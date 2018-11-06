# List of encountered problems :

## 06 Novembre 2018 :
1. Everything works at this stage, we can classify with 87% accuracy but there are three problems:
    1.1. The initial weights were imagenet trained weights which is unjustified (Maybe it gives better results)
    1.2. There are 3 channels but are just three times the same thing. After thinking about this, it is equivalent to giving the Neural network more degrees of freedom. We should instead of doing this, add more filters.
    1.3. We CANNOT convert back from mel spectrogram to an audio because we lost the phase. So we will implement a new CNN that takes 2 channels the amplitude and the phase.