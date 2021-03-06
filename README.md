OVERVIEW
--------
This is the source codes for our MCF model

We used the [Cofactor source codes](https://github.com/dawenl/cofactor) (by Liang) to produce results for cofactor model.

We used the WMF implementation at [here](https://github.com/dawenl/cofactor/blob/master/src/content_wmf.py)

We adapted the ranking metrics from Liang at [here](https://github.com/dawenl/cofactor)

Four folders MovieLensRec, TasteProfile, YahooMusic, KickstarterRec contain the preprocessing steps for each of 4 datasets: 
[MovieLens 20M](https://grouplens.org/datasets/movielens/20m/), 
[Taste Profile](https://labrosa.ee.columbia.edu/millionsong/tasteprofile), 
[Yahoo Music Rating R1](https://webscope.sandbox.yahoo.com/catalog.php?datatype=r&did=1) and 
[Kickstarter]()

Something about sourcecodes:
```
cofactor.py is the implementation of cofactor model
```

```
content_wmf.py is the implementation of wmf model
```

Preparing the datasets:
-----------------------

You should download the 4 datasets: [MovieLens 20M](https://grouplens.org/datasets/movielens/20m/), [Taste Profile](https://labrosa.ee.columbia.edu/millionsong/tasteprofile), [Yahoo Music Rating R1](https://webscope.sandbox.yahoo.com/catalog.php?datatype=r&did=1) and [Kickstarter](https://drive.google.com/drive/folders/0B7caFrTiN5gMbkRGQnZ6R1BzQ0E?usp=sharing) before hand.

Then decompress the data.

For example, with TasteProfile dataset, after downloading it, make a `data/rec_data/all` path in TasteProfile folder and extract the `train_triplets.txt` into it.
The file path will be `TasteProfile/data/rec_data/all/train_triplets.txt`

Running:
------------------------------------------
0. copy the mcf source code into each data processing folder.

For example:
```
cp *.py TasterProfile/
```

1. Preprocess the data by running the script mcf_preprocess.py

This step will generate train/validation/test sets for measuring the model performances
```
python mcf_preprocess.py
```

2. Produce the liked co-occurrence (liked user-user co-occurrence and liked item-item co-occurrence)
```
python produce_positive_cooccurrence.py
```

3. Produce the disliked co-occurrence (disliked item-item co-occurrence)
```
python produce_negative_cooccurrence.py
```

4. Run the MCF model:
```
python mcf_rec.py 0 mcf 1 > mcf.log
```

5. Run the cofactor model:
```
python mcf_rec.py 0 cofactor 1 > cofactor.log
```

6. Run the WMF model:
```
python WMF.py > wmf.log
```
