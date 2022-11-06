def main(dataset_dir):
    ##---------------------------------
    dataset_dir     =   args.dataset_dir
    operation_type  =   args.operation_type
    ga_method       =   args.ga_method
    label_name      =   args.label_name
    save_dir        =   args.save_dir
    verbose         =   args.verbose 
    ##---------------------------------

    ## Load dataset (.csv) file
    df = pd.read_csv(dataset_dir)
    ## process df 
    df.drop(df.columns[[0, 1, 48, 53]], axis=1, inplace=True)
    _dict = {'Gender':{'Male':1, 'Female':0}}  # label = column name
    df.replace(_dict, inplace = True) 
    
    # declare a label dictionary
    label_dict = {"SYS BP": 48, "DYS BP": 49}
    list_label = list(label_dict.values())
    label_idx = label_dict[label_name]
    
    # check the operation type
    if operation_type == "Without-GA": 
        
        ## =========================== Standariztion ==========================
        ## Standard scaler
        Xorg = df.to_numpy()  # Take one dataset

        scaler = StandardScaler()
        Xscaled = scaler.fit_transform(Xorg)
        ## store these off for predictions with unseen data
        Xmeans = scaler.mean_
        Xstds = scaler.scale_

        y = Xscaled[:, label_idx] # SYS (one paramaeter BP)
        X = np.delete(Xscaled, list_label, axis=1) # delete both labels: 'SYS' & 'DYS'

        ## =========================== Apply 10-fold Cross validation ==========================
        ## Load DNN model
        # model = DNN(Xscaled) 
        model = DNN_3Layers(Xscaled)
        n_splits = config.N_SPLITS
        std_r2 = []
        std_mae = []
        cv_set = np.repeat(-1.,Xscaled.shape[0])
        skf = KFold(n_splits = n_splits ,shuffle=True, random_state=42)
        for train_index,test_index in skf.split(Xscaled, y):
            x_train,x_test = Xscaled[train_index],Xscaled[test_index]
            y_train,y_test = y[train_index],y[test_index]
            if x_train.shape[0] != y_train.shape[0]:
                raise Exception()
            # model.fit(x_train,y_train)

            # ----------------------------------------
            callback = keras.callbacks.EarlyStopping(monitor='loss', patience=5) 

            model.fit(x_train,y_train, epochs=config.NUM_EPOCHS, batch_size=config.BATCH_SIZE,
                                shuffle=True,  callbacks=[callback],  validation_split=0, verbose=verbose)
            # ----------------------------------------

            predicted_y = model.predict(x_test)
            LOG_INFO(f"Individual R = {pearsonr(y_test, predicted_y)}", mcolor="green")
            cv_set[test_index] = predicted_y[:, 0]

            ###### Load std of r2 and mae
            std_mae.append(metrics.mean_absolute_error((y_test * Xstds[label_idx]) + Xmeans[label_idx], (predicted_y * Xstds[label_idx]) + Xmeans[label_idx]))
            std_r2.append(metrics.r2_score((y_test * Xstds[label_idx]) + Xmeans[label_idx], (predicted_y * Xstds[label_idx]) + Xmeans[label_idx]))
                
        LOG_INFO(f"====> Overall R   = {pearsonr(y,cv_set)}", mcolor="green")

        ## For Get real values of target label
        y = (y * Xstds[label_idx]) + Xmeans[label_idx]
        cv_set = (cv_set * Xstds[label_idx]) + Xmeans[label_idx]

        ### =============== Measure all indices ================================
        LOG_INFO(f"====> Overall R   = {pearsonr(y,cv_set)}", mcolor="red")
        LOG_INFO(f"====> R^2 Score   = {metrics.r2_score(y, cv_set)}", mcolor="red")
        LOG_INFO(f"====> MAE         = {metrics.mean_absolute_error(y, cv_set)}", mcolor="red")
        LOG_INFO(f"====> MSE         = {metrics.mean_squared_error(y, cv_set)}", mcolor="red")
        LOG_INFO(f"====> RMSE        = {rmse(y, cv_set)}", mcolor="red")
        LOG_INFO(f"====> MSLE        = {metrics.mean_squared_log_error(y, cv_set)}", mcolor="red")
        LOG_INFO(f"====> EVS         = {metrics.explained_variance_score(y, cv_set)}", mcolor="red")

        ### =============== plot some measurement graphs ==============================
        ## Bland-altman graph
        bland_altman_plot(y, cv_set, label_name, save_dir, operation_type, verbose)

        ## Plot estimated and predicted 
        r2 = metrics.r2_score(y, cv_set)
        mae = metrics.mean_absolute_error(y, cv_set)
        act_pred_plot(y, cv_set, label_name, save_dir, [mae, r2], [std_mae, std_r2], operation_type, verbose)

    elif operation_type == "With-GA": 
        ## =========================== Standariztion ==========================
        ## Standard scaler
        Xorg = df.to_numpy()  # Take one dataset

        scaler = StandardScaler()
        Xscaled = scaler.fit_transform(Xorg)
        ## store these off for predictions with unseen data
        Xmeans = scaler.mean_
        Xstds = scaler.scale_

        y = Xscaled[:, label_idx] # SYS (one paramaeter BP)
        X = np.delete(Xscaled, list_label, axis=1) # delete both labels: 'SYS' & 'DYS'

        ## =========================== GA selection =============================
        ## check chosing ga
        if ga_method == "filter":
            ## filter method
            fsga = Feature_Selection_GA_Filter(X,y, verbose=verbose) 
            pop = fsga.generate(config.POPULATION_SIZE, config.GENERATION_SIZE) 
            pp = fsga.plot_feature_set_score(config.GENERATION_SIZE) 
            print("Best Indices: " +str(pop))

        elif ga_method == "wrap":
            ## load a model for wrap ga method
            ga_model = SVR(kernel='rbf') # you can load any model
            ## wrap method
            fsga = Feature_Selection_GA_Wrap(X,y, ga_model, verbose=verbose)
            pop = fsga.generate(config.POPULATION_SIZE, config.GENERATION_SIZE) 
            pp = fsga.plot_feature_set_score(config.GENERATION_SIZE) 
            print("Best Indices: " +str(pop))


        ## =================== get the selected features index through GA =============
        get_best_ind = []
        for i in range(len(pop)):
            if pop[i] == 1:
                get_best_ind.append(i)
                
        X_selct = X[:, get_best_ind]
        print(X_selct.shape)  

        ## =========================== Apply 10-fold Cross validation ==========================
        ## Load DNN model
        # model = DNN(X_selct) 
        model = DNN_3Layers(X_selct)
        n_splits = config.N_SPLITS
        std_r2 = []
        std_mae = []
        cv_set = np.repeat(-1.,X_selct.shape[0])
        skf = KFold(n_splits = n_splits ,shuffle=True, random_state=42)
        for train_index,test_index in skf.split(X_selct, y):
            x_train,x_test = X_selct[train_index],X_selct[test_index]
            y_train,y_test = y[train_index],y[test_index]
            if x_train.shape[0] != y_train.shape[0]:
                raise Exception()
            # model.fit(x_train,y_train)

            # ----------------------------------------
            callback = keras.callbacks.EarlyStopping(monitor='loss', patience=5)                                    

            model.fit(x_train,y_train, epochs=config.NUM_EPOCHS, batch_size=config.BATCH_SIZE,
                                shuffle=True, callbacks=[callback], validation_split=0, verbose=verbose)
            # ---------------------------------------

            predicted_y = model.predict(x_test)
            LOG_INFO(f"Individual R = {pearsonr(y_test, predicted_y)}", mcolor="green")
            cv_set[test_index] = predicted_y[:, 0]

            ###### Load std of r2 and mae
            std_mae.append(metrics.mean_absolute_error((y_test * Xstds[label_idx]) + Xmeans[label_idx], (predicted_y * Xstds[label_idx]) + Xmeans[label_idx]))
            std_r2.append(metrics.r2_score((y_test * Xstds[label_idx]) + Xmeans[label_idx], (predicted_y * Xstds[label_idx]) + Xmeans[label_idx]))
                
        LOG_INFO(f"====> Overall R   = {pearsonr(y,cv_set)}", mcolor="green")

        ## For Get real values of target label
        y = (y * Xstds[label_idx]) + Xmeans[label_idx]
        cv_set = (cv_set * Xstds[label_idx]) + Xmeans[label_idx]

        ### =============== Measure all indices ================================
        LOG_INFO(f"====> Overall R   = {pearsonr(y,cv_set)}", mcolor="red")
        LOG_INFO(f"====> R^2 Score   = {metrics.r2_score(y, cv_set)}", mcolor="red")
        LOG_INFO(f"====> MAE         = {metrics.mean_absolute_error(y, cv_set)}", mcolor="red")
        LOG_INFO(f"====> MSE         = {metrics.mean_squared_error(y, cv_set)}", mcolor="red")
        LOG_INFO(f"====> RMSE        = {rmse(y, cv_set)}", mcolor="red")
        LOG_INFO(f"====> MSLE        = {metrics.mean_squared_log_error(y, cv_set)}", mcolor="red")
        LOG_INFO(f"====> EVS         = {metrics.explained_variance_score(y, cv_set)}", mcolor="red")

        ### =============== plot some measurement graphs ==============================
        ## Bland-altman graph
        bland_altman_plot(y, cv_set, label_name, save_dir, operation_type, verbose)

        ## Plot estimated and predicted 
        r2 = metrics.r2_score(y, cv_set)
        mae = metrics.mean_absolute_error(y, cv_set)
        act_pred_plot(y, cv_set, label_name, save_dir, [mae, r2], [std_mae, std_r2], operation_type, verbose)