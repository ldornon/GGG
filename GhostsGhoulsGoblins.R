library(tidymodels)
library(vroom)

ggg_train <- vroom("./train.csv") 
ggg_mv_train <- vroom("./trainWithMissingValues.csv")
ggg_test <- vroom("./test.csv")

my_recipe <- recipe(type~., data = ggg_mv_train) %>% 
  step_impute_mean(bone_length,rotting_flesh, hair_length) 

prep <- prep(my_recipe)
baked <- bake(prep, new_data = ggg_mv_train)

rmse_vec(ggg_train[is.na(ggg_mv_train)],baked[is.na(ggg_mv_train)])


