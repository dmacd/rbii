  
x have claude refactor all classes to ClassName so I can actually read this
- can abbreviate DSL



- validate this is actually doing what i want a
  - double check transformer schedule makes sense 
  - can transformers edit old code? no  -- this design is totally deficient
  - can I give claude a pdf for reference to understand the intent? 
  x start eliminating the dumb trivial shortcuts
    - like in character memory lookup:
      - disable use of bigram and trigram shortcuts
      - add in enough that we could actually learn these in the program store 

- write out the actually program trees 
- log shit to aim
  - log incumbents every N steps
  - log full character stream
  - 

    




- what do i really want from an experiment?
  - to show that cool and useful structure gets extracted and encoded when 
    the primitives are high level perception and reasoning modules
  - to show that useful domain extraction can happen in neuro-symbolic contexts
  - to show that language models can be formed this way and automated the 
    search for good primitives
  - 


## next iteration

- dispense with distribution predictions, i want char predictions
- predictor language must also be universal