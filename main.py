print("Starting ...")
from dataReader import DataReader
from utils import extract_config, get_dataset_from_file

# Hyper-parameters
BATCH_SIZE = 32
EPOCHS = 5
NB_ITERATIONS = 1
NB_SAMPLE = None  # integer or None value for all the dataset

# Data to use, Model to run
GET_AND_SAVE_DATA = False  # True, False
SAVE_MODEL = True  # True, False
DATA_TYPE = 'All'  # 'Audio' 'MFCC' 'Text' 'Image' 'All'
MULTI_TYPE = "Early"  # 'Early', 'Late', 'Hybrid', 'Uni'

DATASET_PATH, MODEL_CLASS, MODEL_DIR,  INPUT_SHAPE = extract_config(DATA_TYPE, MULTI_TYPE)

if __name__ == '__main__':
    data = DataReader()

    if GET_AND_SAVE_DATA:
        print("Getting data ...")
        X_train, X_val, X_test, y_train, y_val, y_test = data.get_dataset(DATA_TYPE, NB_SAMPLE)

    if MULTI_TYPE in ['Uni','Late','Hybrid','Early']:        
        print(f"\nPreparing data for {MULTI_TYPE} model training and evaluation...")
        X_train, X_val, X_test, y_train, y_val, y_test = get_dataset_from_file(DATASET_PATH)


        print("Building model ...")
        model = MODEL_CLASS(X_train, X_val, X_test, y_train, y_val, y_test,
                               data.nb_of_label, batch_size=BATCH_SIZE, epochs=EPOCHS, input_shape=INPUT_SHAPE)

        model.build_model()

        model.model_result()

        print("\nTraining model ...")
        model.save_config_to_csv(MODEL_DIR, BATCH_SIZE, EPOCHS, NB_ITERATIONS, DATA_TYPE, DATASET_PATH, MODEL_CLASS,
                                    INPUT_SHAPE)

        for k in range(NB_ITERATIONS):
            print(f"Iteration {k+1}/{NB_ITERATIONS}")
            try:model.load_model(MODEL_DIR)
            except: print(f'Aucun model trouvé, un nouveau model sera sauvegardé sous le nom de : {MODEL_DIR}')
            history = model.fit()
            model.save_history_to_csv(MODEL_DIR, ITERATION=k)
            model.evaluate()         
             
            if SAVE_MODEL : model.save(MODEL_DIR)
        model.plot_confusion_matrix(X_test, y_test)  
        model.plot_tree_model()
        print(model.summary())
        model.classification_report(X_test, y_test, save=True)

    else:
        print(f"Invalid argument for MULTI_TYPE")

print('Done !')
