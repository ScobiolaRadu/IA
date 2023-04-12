# Attempts:

CNN:
1. 10 epochs, batch size 32 -> unsuccesful
2. 30 epochs, batch size 32 -> unsuccesful
3. 10 epochs, batch size 16, learning rate 0.001 -> unsuccesful
4. 2 epochs, batch size 32, learning rate 0.001 -> succesful, score 0.37919
5. 10 epochs, batch size 32, learning rate 0.001 -> succesful, score 0.48908
6. 20 epochs, batch size 32, learning rate 0.001 -> succesful, score 0.4482
7. 10 epochs, batch size 16, learning rate 0.001 -> succesful, score 0.39857
8. 10 epochs, batch size 32, learning rate 0.0005, added layers -> succesful, score 0.52242
9. 10 epochs, batch size 32, learning rate 0.0005, added layer 512 -> succesful, score 0.5062

Random Forests:
1. random state 0 -> validation f1 score: 0.681246874295858
2. random state 1 -> validation f1 score: 0.6831808641241931
3. random state 2 -> validation f1 score: 0.6986317883189681
4. random state 3 -> validation f1 score: 0.70098177650049
5. random state 4 -> validation f1 score: 0.6942491721303189
