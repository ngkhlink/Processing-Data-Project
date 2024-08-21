# Processing Data Project in Python
#1. Import the required libraries

      def warn(*args, **kwargs):
    pass
    import warnings
    warnings.warn = warn

#2. Import pandas, seaborn, numpy, matpotlib, scikit-learn

    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler,PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    %matplotlib inline

#3. Import Datasets

    import piplite
        await piplite.install('seaborn')
    filepath='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv'
    df = pd.read_csv(filepath, header=0)
    df.head()

#4. Display the data types of each column

    df.dtypes
    df.describe()
    
#5. Drop the columns "id" and "Unnamed: 0"

    df.drop(["Unnamed: 0", "id"], axis=1, inplace=True)

#6: Check the missing values

    print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
    print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())

#7: Replace the missing values

    mean=df['bedrooms'].mean()
    df['bedrooms'].replace(np.nan,mean, inplace=True)
    mean=df['bathrooms'].mean()
    df['bathrooms'].replace(np.nan,mean, inplace=True)
    
#8: Count the number of "houses" and unique "floors" value, convert to the dataframe

    df['floors'].value_counts().to_frame()

#9: Visualisation \
Determine whether "houses" with a waterfront view or without a waterfront view have more price outliers

    import matplotlib.pyplot as plt
    import seaborn as sns
    %matplotlib inline
    sns.boxplot(x='waterfront', y='price', data=df)

Determine if the feature "sqft_above" is negatively or positively correlated with price \
Engine size as potential predictor variable of price

    sns.regplot(x="sqft_above", y="price", data=df)
    plt.ylim(0,)

#10. Fit a linear regression model and caculate the R^2

    X = df[['long']]
    Y = df['price']
    lm = LinearRegression()
    lm.fit(X,Y)
    lm.score(X, Y)

#11. Fit a linear regression model to predict the "price" and calculate the R^2

    from sklearn.linear_model import LinearRegression
    
Create the linear regression object

    lm = LinearRegression()
    lm

    X = df[['sqft_living']]
    Y = df['price']
    
Fit the linear model using sqft_living

    lm.fit(X,Y)

Output a prediction
    
    Yhat=lm.predict(X)
    Yhat[0:5]

array([287555.06702451, 677621.82640197, 172499.40418656, 506441.44998452, 427866.85097324])

    lm.intercept_

-43580.74309447396

    lm.coef_

array([280.6235679])

Plugging in the actual values we get:
Price = -43580.743 + 280.623 x sqft_living

  #12. Find the slope and intercept of the model
Slope

    lm1.coef_

array([[280.6235679]])

Intercept

    lm1.intercept_

array([-43580.74309447])

sqft_living

    lm.fit(X, Y)

Find the R^2
    
    print('The R-square is: ', lm.score(X, Y))

The R-square is:  0.4928532179037931 \
  #13. Fit a linear regression model to predict the "price"

    features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()

    X = df[['floors', 'waterfront', 'lat', 'bedrooms', 'sqft_basement', 'view', 'bathrooms', 'sqft_living15', 'sqft_above', 'grade', 'sqft_living']]
    Y = df['price']
    lr.fit(X,Y)
    r2_score = lr.score(X,Y)

    lm.fit(X, Y)

Find the R^2

    print('The R-square is: ', lm.score(X, Y))

The R-square is:  0.6576861682430691 \
  #14. Model Evaluation and Refinement

    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import train_test_split
    print("done")

  done

  Split the data into training and testing sets

    features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]
    X = df[features]
    Y = df['price']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)

    print("number of test samples:", x_test.shape[0])
    print("number of training samples:",x_train.shape[0])

number of test samples: 3242 \
number of training samples: 18371

Create and fit a Ridge regression object

    from sklearn.linear_model import Ridge
    RM = Ridge(alpha=0.1)
    RM.fit(x_train,y_train)
    RM.score(x_test,y_test)

The R-square is: 0.6478759163939112

Perform a second-order polynomial

    pr = PolynomialFeatures(degree=2)
    x_train_pr = pr.fit_transform(x_train)
    x_test_pr = pr.fit_transform(x_test)

    RM = Ridge(alpha=0.1)

    RM.fit(x_train_pr, y_train)
    RM.score(x_test_pr, y_test)

The R-square is: 0.7002744273539745
