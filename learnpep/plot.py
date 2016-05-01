__description__ = \
"""
Standard plots for output of machine learning runs.
"""
__author__ = "Michael J. Harms"
__date__ = "2016-04-23"

import numpy as np
import matplotlib.pylab as plt
from matplotlib.backends.backend_pdf import PdfPages

def correlation(ml_data,ml_machine,pdf_file=None,max_value=12):

    if pdf_file != None:
        pdf = PdfPages(pdf_file)
    
    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(7, 7)) 
    
    # Plot sundry dashed lines
    plt.plot(np.array([-1*max_value,max_value]),np.array([-1*max_value,max_value]),'k--')
    plt.plot((0,0),(-1*max_value,max_value),'k--')
    plt.plot((-1*max_value,max_value),(0,0),'k--')
    
    # Plot training set
    train_prediction = ml_machine.predict(ml_data.training_features)
    plt.plot(ml_data.training_values,
             train_prediction,
             "o",color="red")

    # Plot test set
    test_prediction = ml_machine.predict(ml_data.test_features)
    plt.plot(ml_data.test_values,
             test_prediction,
             "o",color="blue")

    
    m, b = np.polyfit(ml_data.test_values, test_prediction, 1)
    x = np.array((-1*max_value,max_value))
    plt.plot(x, m*x + b, 'k-',linewidth=2.9)
    
    plt.ylim(-1*max_value,max_value)
    plt.xlim(-1*max_value,max_value)

    plt.xlabel("measured value")
    plt.ylabel("predicted value")
    
    if pdf_file != None:
        pdf.savefig()
        pdf.close()


