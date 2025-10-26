# Voice Impersonation Attack Framework (VIAF)

> **Boson Hackathon 2025**: Can we fool speaker recognition systems with AI-generated voices? Let's find out! 🎤🤖

## 🎯 What This Repo Is About

This is a hackathon project that explores whether we can use Boson's Higgs Audio v2 to generate voices that can trick speaker recognition systems. Think of it as a "voice security stress test" - we're basically trying to break voice authentication systems to see how robust they really are.

## 🔬 What We're Testing

### The Big Questions
We're basically asking:
- **Can AI voices fool speaker recognition?** - How good are we at making fake voices that sound real?
- **Which systems are easiest to trick?** - Some speaker recognition systems might be more vulnerable than others
- **What makes a good voice clone?** - Different prompts and settings might work better
- **How do we measure success?** - We need good ways to tell if our fake voices are convincing

### Our Approach
We're running two main types of experiments:

#### 1. Prompt Testing
- **Testing 100+ different prompts** to see which ones work best for voice cloning
- **Using different sentences** to see if some are easier to clone than others
- **Measuring how similar** our fake voices are to the real ones
- **Testing across different speakers** to see if some voices are easier to clone

#### 2. Parameter Tuning
- **Playing with generation settings** like temperature, top_p, and top_k
- **Finding the sweet spot** for making voices that sound most like the target
- **Testing different combinations** to see what works best
- **Cross-checking our results** with multiple voice analysis methods

## 🏗️ What's In This Repo

### The Core Stuff
- **`src/voice_generator.py`** - The main engine that generates voices using Higgs Audio v2
- **`src/attack_strategies.py`** - Different ways to try to fool speaker recognition systems
- **`src/voice_analyzer.py`** - Tools to analyze and compare voices
- **`src/embedding_scorer.py`** - Scoring how similar voices are to each other

### The Experiments
- **`experiments/automated_prompt_experiment/`** - Testing tons of different prompts automatically
- **`experiments/decoding_grid_experiment/`** - Finding the best generation parameters
- **`experiments/full_eval/`** - Running comprehensive tests and analyzing results

### The Demo Apps
- **`app/streamlit_app.py`** - A web app to try out voice cloning
- **`app_legacy/`** - Some older demo code and examples

## 📊 What We Found

### The Good News (for attackers 😈)
- **We can get really close!** - Some of our fake voices are 99%+ similar to the real ones
- **Different prompts matter a lot** - Some work way better than others
- **Parameters make a huge difference** - The right settings can dramatically improve results
- **Some speakers are easier to clone** - Not all voices are equally hard to fake

### The Numbers
- **Best similarity scores**: Up to 0.92+ on WavLM cosine similarity
- **Best parameters**: temperature=0.7, top_p=0.95, top_k=50 (usually)
- **Best prompts**: Neutral prompts + certain sentences work best
- **Success rates**: We can fool some systems pretty consistently

## 🎯 Why This Matters

### For Security
- **Voice authentication might not be as secure** as we thought
- **Different systems have different vulnerabilities** - some are easier to fool
- **We can measure and quantify these risks** - it's not just theoretical

### For AI Development
- **Voice generation is getting really good** - the quality is impressive
- **We can optimize for specific goals** - like making voices that fool recognition systems
- **There are measurable ways to improve** - it's not just trial and error

## 🛠️ The Tech Stack

### What We Used
- **Boson's Higgs Audio v2** - For generating the voices
- **Multiple speaker recognition systems** - To test against different approaches
- **WavLM, ECAPA-TDNN, MFCC** - Different ways to analyze and compare voices
- **Python + lots of ML libraries** - For all the analysis and experiments

### How We Built It
- **Modular design** - Easy to add new attack strategies or test new systems
- **Automated experiments** - We can test hundreds of combinations automatically
- **Comprehensive metrics** - Multiple ways to measure success
- **Easy to extend** - Add new speakers, systems, or attack methods

## 🚀 What's Next

### Potential Improvements
- **More attack strategies** - There are probably other ways to fool these systems
- **Better optimization** - We could probably get even better results
- **More speaker recognition systems** - Test against more targets
- **Real-time attacks** - Can we fool systems in real-time?

### Research Directions
- **Defense mechanisms** - How can we make these systems more robust?
- **Detection methods** - Can we tell when a voice is AI-generated?
- **Better metrics** - More sophisticated ways to measure voice similarity
- **Cross-domain attacks** - What about fooling other types of voice systems?

## 📚 Files You Might Care About

- **`PROJECT_SPEC.md`** - More detailed technical specs
- **`experiments/results/`** - All our experimental results and findings
- **`app/`** - Demo applications to try out the voice cloning
- **`src/`** - The core code for voice generation and analysis

## 🤝 Contributing

This is a hackathon project, but if you want to:
- **Add new attack strategies** - Go for it!
- **Test against more systems** - The more the merrier
- **Improve the metrics** - Better ways to measure success
- **Fix bugs** - There are probably some 😅

## 📄 License

MIT License - feel free to use this for your own voice security research!

## 🙏 Thanks

- **Boson AI** for the awesome Higgs Audio v2 API
- **The speaker recognition community** for all the open-source tools
- **Everyone who helped** with this hackathon project

---

**Built for Boson Hackathon 2025** 🚀

*Can we break voice authentication? Spoiler: Yes, we can! 😎*