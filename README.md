# ELO Score Task Organizer

Sort tasks by priority using smaller lists. E.g., you have 270 tasks (unreasonable) to sort, so you sort them within batches of 20 tasks at a time. The app uses an ELO scoring system to adjust task priorities based on your rankings, so over time, the tasks you care about rise to the top.

## Installation

```bash
uv sync
streamlit run main.py
```

## 1. Configurable Batch Size

You can now customize the number of tasks shown in each ranking batch:

- **Location**: Sidebar → ⚙️ Configuration → Batch Size slider
- **Range**: 3-20 tasks per batch
- **Default**: 10 tasks
- **Usage**: Smaller batches are quicker to rank but require more rounds; larger batches give more comprehensive comparisons

## 2. LLM Auto-Sort Feature (Enhanced with Few-Shot Learning)

Automatically sort tasks using AI (OpenRouter API) with intelligent few-shot learning:

### Setup:
1. Go to Sidebar → ⚙️ Configuration → 🤖 LLM Auto-Sort
2. Check "Enable LLM Auto-Sort"
3. Enter your OpenRouter API key (get one from [openrouter.ai](https://openrouter.ai))
4. Select your preferred AI model.

### How It Works:
- **Few-Shot Learning**: The AI learns from your previous ranking decisions
- Uses the last 3 batches you manually ranked as examples
- This helps the AI understand your prioritization preferences
- As you rank more batches, the AI gets better at matching your priorities

### Usage:
- Once enabled, a "🤖 Auto-Sort with AI" button appears at the top of the ranking interface
- Click it to have AI automatically analyze and rank your current batch
- The AI considers:
  - Task titles and descriptions
  - Your previous ranking patterns (few-shot examples)
  - Context from historically ranked tasks
- Rankings are automatically applied and saved
- Next batch loads automatically

### Benefits:
- Faster than manual drag-and-drop ranking
- **Learns your prioritization style** over time
- Consistent logic informed by your past decisions
- Works great for large task lists
- Can process batches in seconds
