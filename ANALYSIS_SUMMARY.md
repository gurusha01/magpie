# Analysis System - Quick Reference

## What I Created

I've created a comprehensive analysis system for evaluating your multi-agent negotiation simulations. Here's what's included:

### 1. Main Analysis Script: `analysis.py`

A Python script that evaluates simulations on 5 key dimensions:

1. **Privacy Preservation (Implicit)** - Checks if agents leaked private info under implicit guidelines
2. **Privacy Preservation (Explicit)** - Checks if agents leaked private info under explicit guidelines
3. **Task Completion** - Evaluates consensus achievement, deliverable correctness, and success criteria
4. **Bad Behaviors** - Detects sycophancy, lying, threatening, manipulation, shaming, power-seeking
5. **Good Behaviors** - Detects compromise and collaboration

### 2. Unified LLM Interface

The `generate_llm_response()` function supports **5 LLM providers**:

- ✅ **Gemini** (Google)
- ✅ **OpenAI** (GPT-4, GPT-4o, etc.)
- ✅ **Together AI** (Open-source models)
- ✅ **AWS Bedrock** (Claude, Llama, Titan, AI21) ← **NEWLY ADDED**
- Easy to extend for more providers

### 3. Supporting Files

- **README_analysis.md** - Complete documentation
- **AWS_BEDROCK_SETUP.md** - Step-by-step AWS Bedrock setup guide
- **run_analysis.sh** - Batch processing shell script
- **ANALYSIS_SUMMARY.md** - This file

## Quick Start

### Installation

```bash
# Install dependencies
pip install openai google-genai boto3

# Set up API keys in .env file (see README_analysis.md)
```

### Basic Usage

```bash
# Analyze a single simulation
python analysis.py \
  --scenario academic_job \
  --simulation simulations/explicit-data2/gemini/sim_academic_job.json \
  --guideline explicit

# Analyze all simulations
python analysis.py --all

# Using different LLMs
python analysis.py --all --llm gemini --model gemini-2.0-flash-exp
python analysis.py --all --llm openai --model gpt-4o
python analysis.py --all --llm bedrock --model anthropic.claude-3-5-sonnet-20241022-v2:0
```

### Using the Shell Script

```bash
# Make it executable (if not already)
chmod +x run_analysis.sh

# Analyze all simulations
./run_analysis.sh --all

# Analyze with AWS Bedrock
./run_analysis.sh --all --llm bedrock --model anthropic.claude-3-haiku-20240307-v1:0

# Analyze only explicit guideline simulations
./run_analysis.sh --explicit-only

# Analyze only implicit guideline simulations
./run_analysis.sh --implicit-only
```

## Output Format

Results are saved as JSON files in `analysis/` directory:

```
analysis/
├── academic_job_gemini_explicit_analysis.json
├── academic_job_gemini_implicit_analysis.json
├── allocation_gpt5_explicit_analysis.json
└── ...
```

Each file contains:

```json
{
  "metadata": { ... },
  "privacy_preservation": {
    "Agent Name": {
      "private_info_items": [...],
      "privacy_score": 8
    }
  },
  "task_completion": {
    "consensus_achieved": true,
    "overall_success": true,
    "task_completion_score": 9
  },
  "bad_behaviors": {
    "sycophancy": { "detected": false, ... },
    "lying": { "detected": false, ... },
    ...
  },
  "good_behaviors": {
    "compromise": { "detected": true, "quality": "high", ... }
  }
}
```

## LLM Provider Comparison

| Provider | Best For | Cost | Speed | Setup Difficulty |
|----------|----------|------|-------|-----------------|
| **Gemini** | General use, fast iteration | Low | Fast | Easy |
| **OpenAI** | High-quality analysis | Medium-High | Medium | Easy |
| **Together AI** | Open-source models, flexibility | Low-Medium | Medium-Fast | Easy |
| **AWS Bedrock** | Enterprise, Claude access | Medium | Medium | Medium |

## Recommended Workflow

### For Testing/Development
```bash
# Use Gemini Flash for speed and low cost
python analysis.py --all --llm gemini --model gemini-2.0-flash-exp
```

### For Production Analysis
```bash
# Use Claude 3.5 Sonnet for highest quality
python analysis.py --all --llm bedrock --model anthropic.claude-3-5-sonnet-20241022-v2:0

# Or GPT-4o for balanced performance
python analysis.py --all --llm openai --model gpt-4o
```

### For Budget-Conscious Analysis
```bash
# Use Claude 3 Haiku via Bedrock (very cost-effective)
python analysis.py --all --llm bedrock --model anthropic.claude-3-haiku-20240307-v1:0

# Or Gemini Flash
python analysis.py --all --llm gemini --model gemini-2.0-flash-exp
```

## AWS Bedrock Highlights

### Supported Model Families

1. **Anthropic Claude** (Recommended)
   - `anthropic.claude-3-5-sonnet-20241022-v2:0` - Best overall
   - `anthropic.claude-3-haiku-20240307-v1:0` - Most cost-effective
   - `anthropic.claude-3-opus-20240229-v1:0` - Most capable

2. **Meta Llama**
   - `meta.llama3-70b-instruct-v1:0`
   - `meta.llama3-8b-instruct-v1:0`

3. **Amazon Titan**
   - `amazon.titan-text-premier-v1:0`
   - `amazon.titan-text-express-v1`

4. **AI21 Labs**
   - `ai21.jamba-instruct-v1:0`
   - `ai21.j2-ultra-v1`

### Setup Steps

1. Install boto3: `pip install boto3`
2. Configure AWS CLI: `aws configure`
3. Request model access in AWS Console (Bedrock → Model access)
4. Run analysis!

See **AWS_BEDROCK_SETUP.md** for detailed instructions.

## Extending the System

### Adding a New LLM Provider

Edit `analysis.py` and add to `generate_llm_response()`:

```python
elif llm_type == "your_provider":
    # Your implementation here
    client = YourProvider(api_key=api_key)
    response = client.generate(prompt)
    return response.text
```

### Adding a New Analysis Dimension

1. Create a new analysis function in `analysis.py`:
   ```python
   def analyze_your_dimension(conversation_log, scenario_data, llm_type, model_name):
       # Your analysis logic
       return analysis_results
   ```

2. Add it to `analyze_simulation()`:
   ```python
   analysis_results["your_dimension"] = analyze_your_dimension(...)
   ```

## File Structure

```
mpi/
├── analysis.py                      # Main analysis script ⭐
├── run_analysis.sh                  # Batch processing script
├── README_analysis.md               # Full documentation
├── AWS_BEDROCK_SETUP.md            # Bedrock setup guide
├── ANALYSIS_SUMMARY.md             # This file
├── data2/                          # Scenario definitions
│   ├── academic_job.json
│   └── ...
├── simulations/                    # Simulation results
│   ├── implicit-data2/
│   │   ├── gemini/
│   │   └── gpt5/
│   └── explicit-data2/
│       ├── gemini/
│       └── gpt5/
└── analysis/                       # Analysis results (created)
    ├── academic_job_gemini_explicit_analysis.json
    └── ...
```

## Common Commands

```bash
# Help
python analysis.py --help
./run_analysis.sh --help

# Analyze everything
python analysis.py --all

# Analyze specific simulation
python analysis.py \
  --scenario academic_job \
  --simulation simulations/explicit-data2/gemini/sim_academic_job.json \
  --guideline explicit

# Use specific LLM
python analysis.py --all --llm bedrock --model anthropic.claude-3-haiku-20240307-v1:0

# Custom output directory
python analysis.py --all --output-dir my_results
```

## Troubleshooting

See **README_analysis.md** for detailed troubleshooting.

Quick fixes:
- **Import errors**: Install missing packages (`pip install openai google-genai boto3`)
- **API key errors**: Check `.env` file or run `aws configure`
- **Model access errors**: Request access in AWS Console (for Bedrock)
- **JSON parsing errors**: Lower temperature or try different model

## Next Steps

1. ✅ Set up your preferred LLM provider
2. ✅ Run a test analysis on one simulation
3. ✅ Review the output JSON
4. ✅ Run batch analysis on all simulations
5. ✅ Aggregate results for your research

## Support

For questions or issues:
1. Check **README_analysis.md** for detailed documentation
2. Check **AWS_BEDROCK_SETUP.md** for Bedrock-specific help
3. Review the prompts in `analysis.py` to understand evaluation criteria
4. Adjust prompts or add new analysis dimensions as needed

---

**Created**: 2025-09-30
**Version**: 1.0
**Author**: AI Assistant (via Cursor)
**Purpose**: Multi-agent simulation analysis with multi-LLM support
