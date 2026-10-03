import pandas as pd

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

df = pd.read_csv("data/monthly_sales.csv")

X = df["previous_month_sales"].values
y = df["sales_revenue"].values


# --------------------------------------------------
# 2. Scale data
# --------------------------------------------------

X = X / 100000
y = y / 100000


# --------------------------------------------------
# 3. Initial parameters
# --------------------------------------------------

theta0 = 0.0
theta1 = 1.0


# --------------------------------------------------
# 4. Prediction function
# --------------------------------------------------

def predict(x):
    return theta0 + theta1 * x


# --------------------------------------------------
# 5. Cost function
# --------------------------------------------------

def cost_function(X, y):
    m = len(X)
    total_error = 0

    for i in range(m):
        prediction = predict(X[i])
        error = prediction - y[i]
        total_error += error ** 2

    return total_error / (2 * m)


# --------------------------------------------------
# 6. Compute gradients
# --------------------------------------------------

def compute_gradients(X, y):
    m = len(X)

    gradient_theta0 = 0
    gradient_theta1 = 0

    for i in range(m):
        prediction = predict(X[i])
        error = prediction - y[i]

        gradient_theta0 += error
        gradient_theta1 += error * X[i]

    gradient_theta0 = gradient_theta0 / m
    gradient_theta1 = gradient_theta1 / m

    return gradient_theta0, gradient_theta1


# --------------------------------------------------
# 7. Gradient descent
# --------------------------------------------------

def gradient_descent(X, y, learning_rate, iterations):
    global theta0, theta1

    for i in range(iterations):

        g0, g1 = compute_gradients(X, y)

        theta0 = theta0 - learning_rate * g0
        theta1 = theta1 - learning_rate * g1

        if i % 100 == 0:
            cost = cost_function(X, y)

            print(
                "Iteration:",
                i,
                "Cost:",
                cost
            )

    return theta0, theta1


# --------------------------------------------------
# 8. Cost before training
# --------------------------------------------------

print("Initial cost:", cost_function(X, y))


# --------------------------------------------------
# 9. Train model
# --------------------------------------------------

theta0, theta1 = gradient_descent(
    X,
    y,
    learning_rate=0.01,
    iterations=1000
)


# --------------------------------------------------
# 10. Results
# --------------------------------------------------

print("\nFinal theta0:", theta0)
print("Final theta1:", theta1)
print("Final cost:", cost_function(X, y))


# --------------------------------------------------
# 11. Test prediction
# --------------------------------------------------

previous_month_sales = 300000

scaled_input = previous_month_sales / 100000

scaled_prediction = predict(scaled_input)

predicted_sales = scaled_prediction * 100000

print(
    "\nIf previous month sales are:",
    previous_month_sales
)

print(
    "Predicted next month sales:",
    predicted_sales
)