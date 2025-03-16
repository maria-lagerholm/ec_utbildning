# Handwritten Arithmetic Solver  

This project recognizes handwritten **digits** and **operators** (`+`, `-`) and solves simple math equations.  


1. I tried **Logistic Regression, SVM, and CNN** on the **MNIST dataset**.  
2. **CNN performed best**, so I used it for the final model.  
3. I augmented and expanded the dataset by adding **synthetic handwritten + and - signs** as well as some varriations to the digits positions - rotation, size, etc.  

## How it works  

- Users **draw an equation** (e.g., `12 + 34`) and click the **Solve** button.  
- The model **recognizes digits and operators**.  
- The app segments the numbers and the operators, then **calculates the result** and displays it.  

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

- Thanks to OpenAI. This project would have taken much longer to complete without their generative models.  
- Thanks to my teacher Antonio Prgomet, as well as the authors of the referenced articles (mentioned in the respective notebooks) and tutorials that provided information and code examples.  