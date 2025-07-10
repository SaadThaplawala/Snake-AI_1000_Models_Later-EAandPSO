🐍 SnakeCrafter: 1000 Models Later – EA and PSO with Neural Network
One script. Multiple strategies. Countless trained snakes.

This project started off with a simple idea — evolve a snake to play the classic Snake game using Neural Networks. But as with most AI things, it quickly spiraled into a lot more.

I present to you SnakeCrafter, a fully self-contained project that combines Evolutionary Algorithms (EA) and Particle Swarm Optimization (PSO) to train neural networks for the snake game. It includes a GUI to visualize gameplay and a headless mode to train faster. The journey was full of tweaks, border-hugging frustrations, and over 1000 trained models later — I learned a lot.

🧠 What’s Inside
AI_project.py: All code lives in one file for easier management and testing.

Two core AI training techniques:

Genetic Algorithm (GA) with mutation, crossover, and survival.

Particle Swarm Optimization (PSO) with velocity and social learning.

Fully working Snake Game GUI for watching trained snakes.

Headless Snake mode for faster training.

Multiple fitness functions — from simple score-and-survive to advanced ones considering wall distance, direction changes, and apple proximity.

Fitness evolution plots and CSVs for all parameter combinations.

Toggle between legacy and extended input states.

⚙️ Fitness Functions (Short Story of Trial & Error)
🔹 Old but Gold
Originally, I used a basic fitness function:

fitness = score * 100 + steps / 10

Worked surprisingly well, but snakes often hugged walls and died in awkward ways.

🔹 Complex Upgrade (a bit too smart?)
Then I added more logic — penalties for being near walls, bonuses for getting closer to the apple, etc. It looked like this:

if score == 0:
  fitness = steps * 0.1
else:
  fitness = (score ** 2) * 100 + steps * 0.5

Bonus/Penalty for wall distance + apple distance delta
It was... complicated. Performance was mixed.

🔹 Model-Specific Tweaks
EA version rewards smooth paths, straight movements, and punishes frequent turns.

PSO version is simpler — focuses more on score and total steps, less on action-based penalties.

📊 Output
After hours of training, I saved:

.pkl files: Best models (some with old, some with new fitness)

.csv files: Tables listing all models, their parameters, and performance

.png plots: Fitness evolution per generation (e.g. best_ga_evolution.png)

🔄 Training Parameters
EA (Genetic Algorithm)
Population sizes from 20 to 200

Crossover types: one_point / two_point

Mutation rates: 0.01 to 0.1

Fitness per individual: average of 3 runs, penalized for short runs or jittery movement

PSO (Particle Swarm Optimization)
Swarm sizes from 20 to 200

w, c1, c2 tuned across wide ranges

More than 300 PSO combinations tested

🕹 How to Use
Run the script:

python AI_project.py

Menu Options
Train new models (GA, PSO, or both)

Test a saved model in GUI

List available saved models

Toggle between extended and legacy input state

Exit

Install Required Packages
pip install pygame numpy matplotlib pandas tqdm joblib

🧪 My Learning Curve (aka Pain)
Initially thought this would be a chill project. Instead:

Wrote everything 3 times. Yep. Started over twice due to weird training crashes and messy code.

Realized complex fitness isn’t always better.

Snakes love hugging walls. Weird.

Models trained over hours — some gave beautiful runs, some kept spinning in circles.

Read lots of other people’s repos just to compare.

But honestly, it was worth it.
This version finally works, and performs well enough to be demoed or trained further.

📁 Files & Folders (Generated After Training)
best_snake_ai_ga_*.pkl — Best GA-trained model

best_snake_ai_pso_*.pkl — Best PSO-trained model

snake_ai_ga_table_*.csv — Full GA model analysis

snake_ai_pso_table_*.csv — Full PSO model analysis

best_ga_evolution.png — GA fitness evolution

best_pso_evolution.png — PSO fitness evolution

🙋‍♂️ Final Thoughts
SnakeCrafter might not be the most optimized snake ever, but it’s mine — built with lots of trial, error, and midnight bug-fixes.

If you're looking to evolve your own snake AI with GA or PSO, feel free to fork, train, and tweak!

Happy crafting! 🐍
