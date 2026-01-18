"""
Task Ranker - ELO-based task prioritization through pairwise comparisons
Run with: streamlit run task_ranker.py
Requires: pip install streamlit streamlit-sortables
"""

import streamlit as st
from streamlit_sortables import sort_items
import re
import json
import math
import random
from openai import OpenAI
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional

# ============================================================================
# Data Models
# ============================================================================


@dataclass
class Task:
    id: str
    title: str
    description: str
    elo: float = 1500.0
    comparisons: int = 0


@dataclass
class AppState:
    tasks: dict = field(default_factory=dict)  # id -> Task
    current_batch: list = field(default_factory=list)  # List of task IDs
    ranking_history: list = field(default_factory=list)
    last_batch_tasks: list = field(default_factory=list)  # For overlap
    file_loaded: bool = False
    # Configuration
    batch_size: int = 10
    use_llm_sort: bool = False
    openrouter_api_key: str = ""
    llm_model: str = "anthropic/claude-4.5-sonnet"


# ============================================================================
# ELO Rating System
# ============================================================================

K_FACTOR = 32  # How much ratings change per comparison


def expected_score(rating_a: float, rating_b: float) -> float:
    """Calculate expected score for player A against player B."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(winner_elo: float, loser_elo: float) -> tuple[float, float]:
    """Update ELO ratings after a comparison. Returns (new_winner_elo, new_loser_elo)."""
    expected_win = expected_score(winner_elo, loser_elo)
    expected_lose = expected_score(loser_elo, winner_elo)

    new_winner = winner_elo + K_FACTOR * (1 - expected_win)
    new_loser = loser_elo + K_FACTOR * (0 - expected_lose)

    return new_winner, new_loser


# ============================================================================
# Markdown Parser
# ============================================================================


def parse_markdown_tasks(content: str) -> list[Task]:
    """Parse markdown file with top-level tasks and nested descriptions."""
    tasks = []
    lines = content.split("\n")

    current_task = None
    current_description_lines = []
    task_counter = 0

    for line in lines:
        # Check for top-level task (# or ## or - [ ] or * at start)
        top_level_match = re.match(r"^(#{1,2}\s+|[-*]\s+\[.\]\s*|[-*]\s+)(.+)$", line)

        if top_level_match:
            # Save previous task if exists
            if current_task:
                current_task.description = "\n".join(current_description_lines).strip()
                tasks.append(current_task)

            # Start new task
            task_counter += 1
            title = top_level_match.group(2).strip()
            current_task = Task(id=f"task_{task_counter}", title=title, description="")
            current_description_lines = []
        elif current_task and line.strip():
            # Add to description if we're in a task
            current_description_lines.append(line)

    # Don't forget the last task
    if current_task:
        current_task.description = "\n".join(current_description_lines).strip()
        tasks.append(current_task)

    return tasks


# ============================================================================
# LLM Auto-Sort Integration
# ============================================================================


def llm_auto_sort(
    tasks: list[Task],
    state: AppState,
    api_key: str,
    model: str = "anthropic/claude-4.5-sonnet",
) -> Optional[list[str]]:
    """Use LLM to automatically sort tasks by priority with few-shot examples. Returns list of task IDs in priority order."""
    if not api_key:
        return None

    # Prepare task list for LLM
    task_descriptions = []
    for i, task in enumerate(tasks, 1):
        desc = f"{i}. {task.title}"
        if task.description:
            desc += f" - {task.description[:200]}"
        task_descriptions.append(desc)

    # Build few-shot examples from ranking history
    few_shot_examples = []
    if state.ranking_history:
        # Get last 3 ranking examples to use as few-shot
        recent_rankings = state.ranking_history[-3:]

        for ranking in recent_rankings:
            if "batch" in ranking:
                batch_task_ids = ranking["batch"]
                # Get tasks with their ELO scores at time of ranking
                example_tasks = []
                for tid in batch_task_ids[:5]:  # Use first 5 for brevity
                    if tid in state.tasks:
                        task = state.tasks[tid]
                        example_tasks.append(f"- {task.title}")

                if example_tasks:
                    few_shot_examples.append(
                        {
                            "tasks": example_tasks,
                            "order": list(range(1, len(example_tasks) + 1)),
                        }
                    )

    # Build messages with few-shot examples
    messages = [
        {
            "role": "system",
            "content": "You are a task prioritization assistant. Analyze tasks and rank them from most important/urgent to least important/urgent. Consider factors like deadlines, impact, dependencies, and urgency.",
        }
    ]

    # Add few-shot examples
    for i, example in enumerate(few_shot_examples, 1):
        example_text = "\n".join(
            [f"{j}. {task}" for j, task in enumerate(example["tasks"], 1)]
        )
        example_order = ",".join(map(str, example["order"]))

        messages.append(
            {
                "role": "user",
                "content": f"Tasks:\n{example_text}\n\nRespond with ONLY the task numbers in priority order, separated by commas.",
            }
        )
        messages.append({"role": "assistant", "content": example_order})

    # Add current task batch
    current_prompt = f"""Tasks:
{chr(10).join(task_descriptions)}

Respond with ONLY the task numbers in priority order, separated by commas. For example: 3,1,5,2,4"""

    messages.append({"role": "user", "content": current_prompt})

    try:
        # Initialize OpenAI client with OpenRouter base URL
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

        response = client.chat.completions.create(
            model=model, messages=messages, temperature=0.3, max_tokens=100
        )

        sorted_indices_str = response.choices[0].message.content.strip()
        # Parse the response (e.g., "3,1,5,2,4")
        sorted_indices = [
            int(x.strip()) - 1
            for x in sorted_indices_str.split(",")
            if x.strip().isdigit()
        ]

        # Map back to task IDs
        if len(sorted_indices) == len(tasks):
            return [tasks[i].id for i in sorted_indices if i < len(tasks)]

        return None
    except Exception as e:
        print(f"LLM sort error: {e}")
        return None


# ============================================================================
# Batch Selection with Overlap
# ============================================================================


def select_batch(state: AppState, batch_size: int = 10) -> list[str]:
    """Select a batch of tasks with overlap from previous batch."""
    all_task_ids = list(state.tasks.keys())

    if len(all_task_ids) <= batch_size:
        return all_task_ids

    # Include 1-2 tasks from last batch for overlap (if available)
    overlap_count = min(2, len(state.last_batch_tasks))
    overlap_tasks = []

    if state.last_batch_tasks and overlap_count > 0:
        available_for_overlap = [t for t in state.last_batch_tasks if t in all_task_ids]
        overlap_tasks = random.sample(
            available_for_overlap, min(overlap_count, len(available_for_overlap))
        )

    # Fill remaining slots with tasks that have fewer comparisons (prioritize less-compared)
    remaining_slots = batch_size - len(overlap_tasks)
    available = [tid for tid in all_task_ids if tid not in overlap_tasks]

    # Sort by comparison count (ascending) to prioritize less-compared tasks
    available.sort(key=lambda tid: state.tasks[tid].comparisons)

    # Take some from least compared, some random for variety
    least_compared = (
        available[: remaining_slots * 2]
        if len(available) > remaining_slots
        else available
    )
    new_tasks = random.sample(least_compared, min(remaining_slots, len(least_compared)))

    batch = overlap_tasks + new_tasks
    random.shuffle(batch)

    return batch


# ============================================================================
# State Persistence
# ============================================================================

SAVE_FILE = Path("task_ranker_state.json")


def save_state(state: AppState):
    """Save state to JSON file."""
    data = {
        "tasks": {tid: asdict(t) for tid, t in state.tasks.items()},
        "current_batch": state.current_batch,
        "ranking_history": state.ranking_history,
        "last_batch_tasks": state.last_batch_tasks,
        "file_loaded": state.file_loaded,
        "batch_size": state.batch_size,
        "use_llm_sort": state.use_llm_sort,
        "openrouter_api_key": state.openrouter_api_key,
        "llm_model": state.llm_model,
    }
    SAVE_FILE.write_text(json.dumps(data, indent=2))


def load_state() -> Optional[AppState]:
    """Load state from JSON file if it exists."""
    if not SAVE_FILE.exists():
        return None

    try:
        data = json.loads(SAVE_FILE.read_text())
        state = AppState()
        state.tasks = {tid: Task(**t) for tid, t in data.get("tasks", {}).items()}
        state.current_batch = data.get("current_batch", [])
        state.ranking_history = data.get("ranking_history", [])
        state.last_batch_tasks = data.get("last_batch_tasks", [])
        state.file_loaded = data.get("file_loaded", False)
        state.batch_size = data.get("batch_size", 10)
        state.use_llm_sort = data.get("use_llm_sort", False)
        state.openrouter_api_key = data.get("openrouter_api_key", "")
        state.llm_model = data.get("llm_model", "anthropic/claude-3.5-sonnet")
        return state
    except Exception:
        return None


# ============================================================================
# Streamlit UI
# ============================================================================


def init_session_state():
    """Initialize session state."""
    if "state" not in st.session_state:
        loaded = load_state()
        st.session_state.state = loaded if loaded else AppState()


def get_sorted_tasks(state: AppState) -> list[Task]:
    """Get tasks sorted by ELO rating (descending)."""
    return sorted(state.tasks.values(), key=lambda t: t.elo, reverse=True)


def main():
    st.set_page_config(page_title="Task Ranker", page_icon="📋", layout="wide")

    init_session_state()
    state = st.session_state.state

    st.title("📋 Task Ranker")
    st.markdown(
        "*Prioritize your tasks through pairwise comparisons using ELO ratings*"
    )

    # Sidebar for file upload and controls
    with st.sidebar:
        st.header("📁 Load Tasks")

        uploaded_file = st.file_uploader(
            "Upload Markdown file",
            type=["md", "txt"],
            help="Upload a markdown file with tasks as top-level items",
        )

        if uploaded_file:
            content = uploaded_file.read().decode("utf-8")
            tasks = parse_markdown_tasks(content)

            if tasks:
                # Preserve existing ELO scores for tasks that already exist
                existing_elos = {
                    t.title: (t.elo, t.comparisons) for t in state.tasks.values()
                }

                state.tasks = {}
                for task in tasks:
                    if task.title in existing_elos:
                        task.elo, task.comparisons = existing_elos[task.title]
                    state.tasks[task.id] = task

                state.file_loaded = True
                state.last_batch_tasks = []  # Clear last batch
                state.current_batch = select_batch(state, batch_size=state.batch_size)
                save_state(state)
                st.success(f"Loaded {len(tasks)} tasks!")
                st.rerun()
            else:
                st.error(
                    "No tasks found in file. Make sure tasks are top-level items (# or - or * )"
                )

        st.divider()

        st.header("⚙️ Configuration")

        # Batch size configuration
        new_batch_size = st.slider(
            "Batch Size",
            min_value=3,
            max_value=20,
            value=state.batch_size,
            help="Number of tasks to compare at once",
        )
        if new_batch_size != state.batch_size:
            state.batch_size = new_batch_size
            save_state(state)

        # LLM Auto-Sort Configuration
        with st.expander("🤖 LLM Auto-Sort", expanded=state.use_llm_sort):
            use_llm = st.checkbox(
                "Enable LLM Auto-Sort",
                value=state.use_llm_sort,
                help="Use AI to automatically sort tasks in batch",
            )

            if use_llm != state.use_llm_sort:
                state.use_llm_sort = use_llm
                save_state(state)

            if use_llm:
                api_key = st.text_input(
                    "OpenRouter API Key",
                    value=state.openrouter_api_key,
                    type="password",
                    help="Get your API key from openrouter.ai",
                )

                if api_key != state.openrouter_api_key:
                    state.openrouter_api_key = api_key
                    save_state(state)

                model = st.selectbox(
                    "LLM Model",
                    options=[
                        "anthropic/claude-4.5-sonnet",
                        "anthropic/claude-4-haiku",
                        "openai/gpt-4",
                        "openai/gpt-3.5-turbo",
                        "google/gemini-pro",
                    ],
                    index=0 if state.llm_model == "anthropic/claude-4.5-sonnet" else 0,
                    help="Select the AI model to use for sorting",
                )

                if model != state.llm_model:
                    state.llm_model = model
                    save_state(state)

        st.divider()

        st.header("🎮 Controls")

        if st.button(
            "🔄 New Batch", use_container_width=True, disabled=not state.file_loaded
        ):
            state.last_batch_tasks = state.current_batch.copy()
            state.current_batch = select_batch(state, batch_size=state.batch_size)
            save_state(state)
            st.rerun()

        if st.button(
            "🗑️ Reset All Ratings",
            use_container_width=True,
            disabled=not state.file_loaded,
        ):
            for task in state.tasks.values():
                task.elo = 1500.0
                task.comparisons = 0
            state.ranking_history = []
            save_state(state)
            st.rerun()

        if st.button("❌ Clear Everything", use_container_width=True):
            st.session_state.state = AppState()
            if SAVE_FILE.exists():
                SAVE_FILE.unlink()
            st.rerun()

        st.divider()

        st.header("📊 Stats")
        if state.tasks:
            total_comparisons = sum(t.comparisons for t in state.tasks.values()) // 2
            st.metric("Total Tasks", len(state.tasks))
            st.metric("Total Comparisons", total_comparisons)

            avg_comparisons = (
                total_comparisons * 2 / len(state.tasks) if state.tasks else 0
            )
            st.metric("Avg Comparisons/Task", f"{avg_comparisons:.1f}")

            # Progress bar - estimate based on comparisons needed for stable ranking
            # Rule of thumb: each task needs ~log2(n) * 2 comparisons for good coverage
            n_tasks = len(state.tasks)
            # Minimum comparisons for reasonable ranking: each task compared ~3-5 times
            target_comparisons_per_task = max(5, int(math.log2(n_tasks + 1) * 2))
            target_total = (n_tasks * target_comparisons_per_task) // 2

            progress = (
                min(1.0, total_comparisons / target_total) if target_total > 0 else 0
            )

            st.markdown("---")
            st.markdown("**Ranking Progress**")
            st.progress(progress)
            st.caption(
                f"{progress * 100:.0f}% complete ({total_comparisons}/{target_total} comparisons)"
            )

    # Main content area
    if not state.file_loaded or not state.tasks:
        st.info("👈 Upload a markdown file with your tasks to get started!")

        st.markdown(
            """
        ### Expected Format
        
        Your markdown file should have tasks as top-level items:
        
        ```markdown
        # Task 1 Title
        Description for task 1 goes here.
        Can be multiple lines.
        
        # Task 2 Title  
        Description for task 2.
        
        - Task 3 Title
        Another way to define tasks
        
        * Task 4 Title
        Yet another way
        ```
        """
        )
        return

    # Two columns: Ranking interface and Current standings
    col1, col2 = st.columns([3, 2])

    with col1:
        st.header("🎯 Rank These Tasks")

        # LLM Auto-Sort button if enabled
        if state.use_llm_sort and state.openrouter_api_key:
            if st.button(
                "🤖 Auto-Sort with AI", use_container_width=True, type="secondary"
            ):
                batch_tasks = [
                    state.tasks[tid]
                    for tid in state.current_batch
                    if tid in state.tasks
                ]
                with st.spinner("AI is analyzing tasks..."):
                    sorted_task_ids = llm_auto_sort(
                        batch_tasks, state, state.openrouter_api_key, state.llm_model
                    )

                    if sorted_task_ids:
                        # Update ELO scores based on AI sorting
                        for i, winner_id in enumerate(sorted_task_ids):
                            for loser_id in sorted_task_ids[i + 1 :]:
                                winner = state.tasks[winner_id]
                                loser = state.tasks[loser_id]

                                new_winner_elo, new_loser_elo = update_elo(
                                    winner.elo, loser.elo
                                )

                                winner.elo = new_winner_elo
                                loser.elo = new_loser_elo
                                winner.comparisons += 1
                                loser.comparisons += 1

                        # Record history
                        state.ranking_history.append(
                            {
                                "timestamp": datetime.now().isoformat(),
                                "batch": sorted_task_ids,
                                "method": "llm_auto",
                            }
                        )

                        # Prepare next batch
                        state.last_batch_tasks = state.current_batch.copy()
                        state.current_batch = select_batch(
                            state, batch_size=state.batch_size
                        )

                        save_state(state)
                        st.success("AI sorting complete! Loading next batch...")
                        st.rerun()
                    else:
                        st.error(
                            "AI sorting failed. Please check your API key and try manual sorting."
                        )

        st.markdown(
            "**Drag to reorder** from most important (top) to least important (bottom)"
        )

        batch_tasks = [
            state.tasks[tid] for tid in state.current_batch if tid in state.tasks
        ]

        # Regenerate batch if it's too small (e.g., after changing batch size)
        expected_batch_size = min(state.batch_size, len(state.tasks))
        if len(batch_tasks) < expected_batch_size:
            state.current_batch = select_batch(state, batch_size=state.batch_size)
            save_state(state)
            st.rerun()

        if not batch_tasks:
            state.current_batch = select_batch(state, batch_size=state.batch_size)
            save_state(state)
            st.rerun()

        # Build items for sortable - use simple display text
        # We'll track by index since sortables may modify text
        original_items = []
        task_id_by_index = {}

        for idx, task in enumerate(batch_tasks):
            desc_preview = (
                task.description[:80] + "..."
                if len(task.description) > 80
                else task.description
            )
            desc_preview = (
                desc_preview.replace("\n", " ") if desc_preview else "No description"
            )
            # Use a unique separator that won't appear in normal text
            display_text = f"{task.title}\n\n{desc_preview}"
            original_items.append(display_text)
            task_id_by_index[idx] = task.id

        # Create reverse mapping: display_text -> task_id
        display_to_task_id = {
            text: task_id_by_index[i] for i, text in enumerate(original_items)
        }

        st.markdown("---")

        # Drag and drop interface
        # Use batch hash as key to force refresh when batch changes
        batch_key = f"task_sorter_{hash(tuple(state.current_batch))}"
        sorted_items = sort_items(original_items, direction="vertical", key=batch_key)

        # Create mapping from sorted items back to task IDs
        # Handle potential text modifications by matching on title prefix
        def find_task_id(item_text):
            # First try exact match
            if item_text in display_to_task_id:
                return display_to_task_id[item_text]
            # Fall back to matching by title (first line)
            item_title = item_text.split("\n")[0].strip()
            for task in batch_tasks:
                if task.title == item_title:
                    return task.id
            # Last resort: fuzzy match on start
            for orig_text, task_id in display_to_task_id.items():
                if orig_text.startswith(item_title) or item_text.startswith(
                    orig_text[:20]
                ):
                    return task_id
            return None

        st.markdown("---")

        # Show detailed view of tasks in current order
        with st.expander("📝 Task Details (click to expand)", expanded=False):
            for i, item in enumerate(sorted_items, 1):
                task_id = find_task_id(item)
                if task_id and task_id in state.tasks:
                    task = state.tasks[task_id]
                    col_info, col_delete = st.columns([5, 1])
                    with col_info:
                        st.markdown(f"**{i}. {task.title}** (ELO: {task.elo:.0f})")
                        if task.description:
                            st.markdown(f"> {task.description}")
                        st.caption(f"Comparisons so far: {task.comparisons}")
                    with col_delete:
                        if st.button(
                            "🗑️", key=f"del_batch_{task_id}", help="Delete this task"
                        ):
                            del state.tasks[task_id]
                            state.current_batch = [
                                tid for tid in state.current_batch if tid != task_id
                            ]
                            state.last_batch_tasks = [
                                tid for tid in state.last_batch_tasks if tid != task_id
                            ]
                            # Regenerate batch if needed
                            if len(state.current_batch) < min(
                                state.batch_size, len(state.tasks)
                            ):
                                state.current_batch = select_batch(
                                    state, batch_size=state.batch_size
                                )
                            save_state(state)
                            st.rerun()
                    st.markdown("---")

        # Submit button
        if st.button("✅ Submit Ranking", use_container_width=True, type="primary"):
            # Convert sorted display items back to task IDs
            sorted_task_ids = []
            for item in sorted_items:
                task_id = find_task_id(item)
                if task_id:
                    sorted_task_ids.append(task_id)

            if len(sorted_task_ids) != len(batch_tasks):
                st.error("Error mapping tasks. Please try refreshing the page.")
            else:
                # Update ELO scores based on pairwise comparisons
                # Higher ranked (earlier in list) beats lower ranked
                for i, winner_id in enumerate(sorted_task_ids):
                    for loser_id in sorted_task_ids[i + 1 :]:
                        winner = state.tasks[winner_id]
                        loser = state.tasks[loser_id]

                        new_winner_elo, new_loser_elo = update_elo(
                            winner.elo, loser.elo
                        )

                        winner.elo = new_winner_elo
                        loser.elo = new_loser_elo
                        winner.comparisons += 1
                        loser.comparisons += 1

                # Record history
                state.ranking_history.append(
                    {"timestamp": datetime.now().isoformat(), "batch": sorted_task_ids}
                )

                # Prepare next batch with overlap
                state.last_batch_tasks = state.current_batch.copy()
                state.current_batch = select_batch(state, batch_size=state.batch_size)

                save_state(state)
                st.success("Rankings recorded! Loading next batch...")
                st.rerun()

    with col2:
        st.header("🏆 Current Rankings")

        sorted_tasks = get_sorted_tasks(state)

        for i, task in enumerate(sorted_tasks, 1):
            # Color code by rank
            if i <= 3:
                medal = ["🥇", "🥈", "🥉"][i - 1]
            else:
                medal = f"**{i}.**"

            elo_display = f"{task.elo:.0f}"
            comparisons_display = f"({task.comparisons})"

            # Highlight if in current batch
            in_batch = task.id in state.current_batch
            marker = " 🎯" if in_batch else ""

            col_rank, col_del = st.columns([6, 1])
            with col_rank:
                st.markdown(f"{medal} {task.title}{marker}")
                st.caption(f"ELO: {elo_display} • Comparisons: {comparisons_display}")
            with col_del:
                if st.button("🗑️", key=f"del_rank_{task.id}", help="Delete task"):
                    del state.tasks[task.id]
                    state.current_batch = [
                        tid for tid in state.current_batch if tid != task.id
                    ]
                    state.last_batch_tasks = [
                        tid for tid in state.last_batch_tasks if tid != task.id
                    ]
                    if len(state.current_batch) < min(
                        state.batch_size, len(state.tasks)
                    ):
                        state.current_batch = select_batch(
                            state, batch_size=state.batch_size
                        )
                    save_state(state)
                    st.rerun()

        # Export option
        st.markdown("---")
        if st.button("📥 Export Rankings", use_container_width=True):
            export_content = "# Task Rankings\n\n"
            export_content += (
                f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n"
            )

            for i, task in enumerate(sorted_tasks, 1):
                # Use Notion-compatible to-do format
                export_content += f"- [ ] {task.title}\n"
                if task.description:
                    desc_clean = task.description.replace("\n", " ")[:200]
                    export_content += f"    {desc_clean}{'...' if len(task.description) > 200 else ''}\n"

            st.download_button(
                "⬇️ Download as Notion To-Do List",
                export_content,
                file_name="task_rankings.md",
                mime="text/markdown",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
