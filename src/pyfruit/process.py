def process(spark, datapath, n_components=2, image_resize=[20,20], verbose=1):
    from .utils import load_filepath, get_dirname_udf, get_name_udf, load_image_preprocess
    from pyspark.sql.functions import udf, col
    from pyspark.ml.linalg import VectorUDT
    import time
    from pandas import Timedelta
    import numpy as np
    from pyspark.ml.feature import StandardScaler, PCA
    load_image_preprocess_udf = udf(lambda img : load_image_preprocess(img, resize_shape=image_resize), VectorUDT())

    if verbose == 1 :
        print('Loading filepath...', end=' ')
    t0 = time.time()
    df = load_filepath(spark, datapath)
    if verbose == 1 :
        delta_t = np.round(time.time()-t0, 0)
        print('Done ! (%s)' %(str(Timedelta(delta_t, 's'))))
    
    
    if verbose == 1 :
        print('Getting filename, category and image...', end=' ')
    t0 = time.time()
    df = df.select(
        'filepath',
        get_name_udf(col("filepath")).alias("filename") , 
        get_dirname_udf(col("filepath")).alias("category") , 
        load_image_preprocess_udf(col("filepath")).alias("image") , 
        )
    if verbose == 1 :
        delta_t = np.round(time.time()-t0, 0)
        print('Done ! (%s)' %(str(Timedelta(delta_t, 's'))))

    
    if verbose == 1 :
        print('Getting persist dataframe...', end=' ')
    t0 = time.time()
    df.persist()
    if verbose == 1 :
        delta_t = np.round(time.time()-t0, 0)
        print('Done ! (%s)' %(str(Timedelta(delta_t, 's'))))
    
    if verbose == 1 :
        print('Collecting number of categories and files...', end=' ')
    t0 = time.time()
    n_categories = df.select('category').distinct().count()
    n_files = df.count()
    files_per_categ = df.groupBy('category').count()
    if verbose == 1 :
        delta_t = np.round(time.time()-t0, 0)
        print('Done ! (%s)' %(str(Timedelta(delta_t, 's'))))

    print('Number of files : %d' %(n_files))
    print('Number of categories : %d' %(n_categories))
    print(files_per_categ)


    if verbose == 1 :
        print('Normalize image...', end=' ')
    t0 = time.time()
    std = StandardScaler(inputCol="image", outputCol="std",
                                  withStd=True, withMean=True)
    model_std = std.fit(df)
    df = model_std.transform(df)
    if verbose == 1 :
        delta_t = np.round(time.time()-t0, 0)
        print('Done ! (%s)' %(str(Timedelta(delta_t, 's'))))

    if verbose == 1 :
        print('Calculating PCA...', end=' ')
    t0 = time.time()
    pca = PCA(k=n_components, inputCol='std', outputCol='pca')
    model_pca = pca.fit(df)

    df = model_pca.transform(df)
    df = df.filter(df.pca.isNotNull())
    if verbose == 1 :
        delta_t = np.round(time.time()-t0, 0)
        print('Done ! (%s)' %(str(Timedelta(delta_t, 's'))))

    return df
