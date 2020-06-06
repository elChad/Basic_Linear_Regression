from sklearn.linear_model import LinearRegression
import random

#create two empty lists
feature_set = []
target_set = []

#get the number of rows for the data set
number_of_rows = 200

#linit number of values in the data set
random_number_limit = 2000

#create the data set

#feature data set
for i in range(0, number_of_rows):
    x = random.randint(0, random_number_limit)
    y = random.randint(0, random_number_limit)
    z = random.randint(0, random_number_limit)

    #create a linear function for the target data set
    function = (10*x) + (2*y) + (3*z)

    #append the data to the lists
    feature_set.append([x,y,z])
    target_set.append(function)

#Create the LR model
model = LinearRegression()
model.fit(feature_set, target_set)

LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

#Create the test data set
test_set = [[8,10,0]] #expected output = (function(8,10,0) = 100
prediction = model.predict(test_set)

print('Predition: '+str(prediction) +'Coefficients: '+str(model.coef_))



