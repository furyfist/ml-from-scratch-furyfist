**Work Flow**
Mail Data --> Data Preprocessing --> Train Test Split --> Logistic Regression Model --> Trained Logistic Regression Model --> when get new Mail --> Verdict Spam or not Spam
![alt text](image.png)


Here's a crisp explanation of separating data as texts and labels:

```python
X = mail_data['Message']  # Input: All email messages
y = mail_data['Category'] # Output: 1 for ham, 0 for spam
```

That's it! This simple separation:
- `X` contains all the email messages we'll analyze
- `y` contains the corresponding labels (1=ham, 0=spam)

This separation is essential because:
1. Machine learning models need input (X) and output (y) clearly separated
2. We'll use X to train the model to predict y
3. Later, we'll use this to train/test split and convert text to numbers




