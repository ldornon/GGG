library(tidymodels)
library(vroom)
library(embed)
library(DataExplorer)
library(GGally)
library(discrim)

ggg_train <- vroom("./train.csv") 
ggg_mv_train <- vroom("./trainWithMissingValues.csv")
ggg_test <- vroom("./test.csv")

my_recipe <- recipe(type~., data = ggg_mv_train) %>% 
  step_impute_mean(bone_length,rotting_flesh, hair_length) 

prep <- prep(my_recipe)
baked <- bake(prep, new_data = ggg_mv_train)

rmse_vec(ggg_train[is.na(ggg_mv_train)],baked[is.na(ggg_mv_train)])

###############

my_mod <- rand_forest(mtry = tune(),
                      min_n = tune(),
                      trees = 1000) %>% 
  set_engine("ranger") %>% 
  set_mode("classification")

My_Recipe <- recipe(type~., data = ggg_train) %>%
  step_other(all_nominal_predictors(), threshold = .001)
prep <- prep(My_Recipe)
baked <- bake(prep, new_data = ggg_train)


GGG_RF_workflow <- workflow() %>% 
  add_recipe(My_Recipe) %>% 
  add_model(my_mod)

tuning_grid <- grid_regular(mtry(range= c(1,length(ggg_train)-1)),
                            min_n(),
                            levels = 5)

folds <- vfold_cv(ggg_train, v=10, repeats = 1)

CV_results <- GGG_RF_workflow %>% 
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(accuracy))

bestTune <- CV_results %>% 
  select_best("accuracy")

final_wf <-GGG_RF_workflow  %>% 
  finalize_workflow(bestTune) %>% 
  fit(data = ggg_train)

Random_forest_preds <-final_wf %>% 
  predict(new_data = ggg_test, type = "class")

Random_preds <-tibble(id =ggg_test$id,
                      type = Random_forest_preds$.pred_class)

vroom_write(x=Random_preds, file="./GGG_RF_Preds.csv", delim=",")


#######
#Neural Networks

nn_recipe <- recipe(type~., data = ggg_train) %>% 
  update_role(id, new_role = "id") %>% 
  step_range(all_numeric_predictors(), min=0, max=1)

nn_model<- mlp(hidden_units= tune(),
               epochs = 50) %>% 
  set_engine("nnet") %>% 
  set_mode("classification")

nn_wf <-workflow() %>% 
  add_recipe(nn_recipe) %>% 
  add_model(nn_model)
folds <- vfold_cv(ggg_train, v = 10, repeats = 1)

nn_tuneGrid<- grid_regular(hidden_units(range = c(1, 50)),
                           levels =10 )
tuned_nn <- nn_wf %>% 
  tune_grid(resamples = folds, 
            grid =nn_tuneGrid, 
            metrics= metric_set(accuracy))
tuned_nn %>% collect_metrics() %>% 
  filter(.metric=="accuracy") %>% 
  ggplot(aes(x= hidden_units, y= mean)) + geom_line()
bestTune <- tuned_nn %>% select_best("accuracy")

final_wf <- nn_wf %>% 
  finalize_workflow(bestTune) %>% 
  fit(data = ggg_train)
NN_preds <-final_wf %>% 
  predict(new_data = ggg_test, type = "class")

NeuralN_preds <-tibble(id =ggg_test$id,
                      type = NN_preds$.pred_class)

vroom_write(x=NeuralN_preds, file="./NN_Preds.csv", delim=",")


#############
#Boosted Trees

install.packages("lightgbm")
library(bonsai)
library(lightgbm)

boost_model <- boost_tree(tree_depth = tune(), 
                          trees = tune(),
                           learn_rate= tune()) %>% 
  set_engine("lightgbm") %>% 
  set_mode("classification")

boost_wf <-workflow() %>% 
  add_recipe(My_Recipe) %>% 
  add_model(boost_model)
folds <- vfold_cv(ggg_train, v = 10, repeats = 1)

boost_tuneGrid<- grid_regular(tree_depth(),
                           trees(),
                           learn_rate(),
                           levels =10 )

tuned_boost <- boost_wf %>% 
  tune_grid(resamples = folds, 
            grid =boost_tuneGrid, 
            metrics= metric_set(accuracy))

bestTune <- tuned_boost %>% select_best("accuracy")

final_wf <- boost_wf %>% 
  finalize_workflow(bestTune) %>% 
  fit(data = ggg_train)

Boost_preds <-final_wf %>% 
  predict(new_data = ggg_test, type = "class")

Boosted_preds <-tibble(id =ggg_test$id,
                       type = Boost_preds$.pred_class)

vroom_write(x=Boosted_preds, file="./Boosted_Preds.csv", delim=",")

#Naive Bayes

My_Recipe <- recipe(type~., data = ggg_train) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>% 
  step_lencode_glm(all_nominal_predictors(), outcome = vars(type))


nb_model <- naive_Bayes(Laplace = tune(), 
                        smoothness = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("naivebayes")

nb_wf <- workflow() %>% 
  add_recipe(My_Recipe) %>% 
  add_model(nb_model)

tuning_grid <- grid_regular(Laplace(),
                            smoothness())

folds <- vfold_cv(ggg_train, v= 5, repeats = 1)


CV_results <- nb_wf %>% 
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(accuracy))

bestTune <- CV_results %>% 
  select_best("accuracy")

final_wf <- nb_wf %>% finalize_workflow(bestTune) %>% 
  fit(data = ggg_train)

NB_preds <- final_wf %>% 
  predict(new_data = ggg_test, type = "class")

Naive_preds <- tibble(id =ggg_test$id,
                      type = NB_preds$.pred_class)
vroom_write(x=Naive_preds, file="./NBPreds.csv", delim=",")



































  