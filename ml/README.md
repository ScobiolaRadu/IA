# Attempts:

CNN:
1. 10 epochs, batch size 32 -> unsuccesful
2. 30 epochs, batch size 32 -> unsuccesful
3. 10 epochs, batch size 16, learning rate 0.001 -> unsuccesful
4. 2 epochs, batch size 32, learning rate 0.001 -> succesful, score 0.37919
5. 10 epochs, batch size 32, learning rate 0.001 -> succesful, score 0.48908
6. 20 epochs, batch size 32, learning rate 0.001 -> succesful, score 0.4482
7. 10 epochs, batch size 16, learning rate 0.001 -> succesful, score 0.39857
8. 10 epochs, batch size 32, learning rate 0.0005, added 256 and 64 neurons dense layers -> succesful, score 0.52242
9. 10 epochs, batch size 32, learning rate 0.0005, added 512 neurons dense layer -> succesful, score 0.5062

Random Forests:
1. Random state 0, max depth 10, 100 estimators -> f1 score: 0.6812
2. Random state 1, max depth 10, 100 estimators -> f1 score: 0.6831
3. Random state 2, max depth 10, 100 estimators -> f1 score: 0.6986
4. Random state 3, max depth 10, 100 estimators -> f1 score: 0.7009
5. Random state 4, max depth 10, 100 estimators -> f1 score: 0.6942
6. Random state 5, max depth 10, 100 estimators -> f1 score: 0.6993
7. Random state 6, max depth 10, 100 estimators -> f1 score: 0.6751
8. Random state 3, max depth 10, 120 estimators -> f1 score: 0.6996
9. Random state 3, max depth 15, 100 estimators -> f1 score: 0.6807
