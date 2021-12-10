def MTFbasedCNNs_estRIRparams(PEs):  
    x_test = PEs
    estT60snr = np.zeros([7, 2])
    
    x_test = x_test.astype('float32') 
    x_test_1 = x_test [0,:]
    x_test_2 = x_test [1,:]
    x_test_3 = x_test [2,:]
    x_test_4 = x_test [3,:]
    x_test_5 = x_test [4,:]
    x_test_6 = x_test [5,:]
    x_test_7 = x_test [6,:]
    
    TAE_1 = x_test_1.reshape((1,300,1))
    TAE_2 = x_test_2.reshape((1,300,1))
    TAE_3 = x_test_3.reshape((1,300,1))
    TAE_4 = x_test_4.reshape((1,300,1))
    TAE_5 = x_test_5.reshape((1,300,1))
    TAE_6 = x_test_6.reshape((1,300,1))
    TAE_7 = x_test_7.reshape((1,300,1))
    
    estT60snr[0,:] = model_1.predict(TAE_1)
    estT60snr[1,:] = model_2.predict(TAE_2)
    estT60snr[2,:] = model_3.predict(TAE_3)
    estT60snr[3,:] = model_4.predict(TAE_4)
    estT60snr[4,:] = model_5.predict(TAE_5)
    estT60snr[5,:] = model_6.predict(TAE_6)
    estT60snr[6,:] = model_7.predict(TAE_7)
        
    return estT60snr
