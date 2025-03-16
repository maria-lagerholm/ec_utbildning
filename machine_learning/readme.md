# Handwritten Arithmetic Solver  

This project recognizes handwritten **digits** and **operators** (`+`, `-`) to solve simple math equations.  

## Model Training  

1. Trained **Logistic Regression, SVM, and CNN** on the **MNIST dataset**.  
2. **CNN performed best**, so I used it for the final model.  
3. Augmented and expanded the dataset by adding **synthetic handwritten + and - signs**.  

## How It Works  

- Users **draw an equation** (e.g., `12 + 34`).  
- The model **recognizes digits and operators**.  
- The app **calculates the result** and displays it.  

## Wake the app up if it's sleeping! It takes some time to load. :) 

https://eq-solver.streamlit.app/

## Run the App locally 

```bash
pip install -r requirements.txt
streamlit run eq_solver.py
```

## Dataset  

- Augmented MNIST dataset for digits.  
- Expanded with synthetic handwritten + and - signs.  

## Acknowledgements  

- Special thanks to OpenAI. This project would have taken much longer to complete without their generative models.  
- Also, thanks to my teacher Antonio Prgomet, the authors of the referenced articles (mentioned in the respective notebooks) and tutorials that provided information and code examples.  