OVERVIEW
--------
MCF model for Kickstarter dataset 
```
KickstarterRec
```
MCF model for MovieLens 20M dataset:
```
MovieLensRec
```

RUNNING RECOMMENDATION:
------------------------------------------
1. Preprocess the data by running the script mcf_preprocess.py
```
python mcf_preprocess.py
```

2. Produce the liked co-occurrence (liked user-user co-occurrence and liked item-item co-occurrence)
```
python produce_positive_cooccurrence.py
```

3. Produce the disliked co-occurrence (disliked item-item co-occurrence)
```
produce_negative_cooccurrence.py
```

4. Run the MCF model:
```
python mcf_rec.py 0 mcf 1
```

5. Run the cofactor model:
```
python mcf_rec.py 0 cofactor 1
```

