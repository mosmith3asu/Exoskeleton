from Modules.Plotting import Simple_Plot
from Modules.Learning import Regressors
from Modules import Old_Data_Handler
from sklearn import preprocessing
# Uncomment selected regression method
REGRESSION_METHOD = 'Guassian'
#REGRESSION_METHOD = 'Linear'

# Import patient trial data
data = Old_Data_Handler.trial_data_MAT()
trial_data1 = data.data_patient1

# Initialize regressor class
regression = Regressors.regression_methods()

if REGRESSION_METHOD == 'Guassian':
    # Find Guassian Process Regression surface and relevant info

    # Full set of data implementation
    regression_data = trial_data1

    # Excluded data if processing is too long
    #first_n_points = 50
    #regression_data = trial_data1[0:first_n_points, :]

    # Scale data
    regression_data = preprocessing.scale(regression_data)

    # Select Parameters
    kernals = regression.gpr_kernals()
    alpha = 1e-8


    reg_surf, gp, score = regression.gpr_surface(regression_data,kernel=kernals[1],
                                                 alpha = alpha, plot_prior=True,debug=True)


    # reg_surf, gp, score = regression.gpr_surface(regression_data, debug=True)

elif REGRESSION_METHOD == 'Linear':
    # Find Linear Regression surface and relevant info
    nth_order_poly = 8
    reg_surf, lreg, score, polyorder = regression.linreg_surface(trial_data1, search_up_to_order=nth_order_poly, print_best=True)
r2,MSE = score
#Configure plot title
title = f'Guassian Process Regression:' \
        f'\n Kernal={kernals[1]} ' \
        f'\nR^2 = {r2} MSE = {MSE}' \
        f'\n {len(regression_data)} datapoints'

# Plot the regression surface
# No plt_components passed as input so it will create a new plt,fig,ax and return as tuple plt_components
plt_components = Simple_Plot.surfplot3D(reg_surf, title)

# Plot the data as scatterplot on the same plot
# Since we want this on same figure, we pass plt_components
Simple_Plot.scatterplot3D(regression_data, title, plt_components=plt_components)

# Unpack and show plot components
plt, fig, ax = plt_components
plt.show()
