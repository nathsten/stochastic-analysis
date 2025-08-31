# stochastic-analysis
My experiment repo for doing stock and portfolio -analysis and algoritmic trading using stochastic and machine learning methods. 


### File and folder overview:

- `AI Trader`:
    
    An attempt to use a XGB Regressor on bundles of historic prices and indicator to predict future prices. 

    I used this to make a trading algorithm that trades a pair of two stocks. It uses mean-variance optimization to find the best hedge strategy for return. 

- `AKER`:

    Finding best portfolio combination of AKER and FRO using MC simulations

- `EQNR`:

    Simulating EQNR price movents correlating drift with lagged oil prices to see if last price change in oil price affects the next price change in EQNR price. 

- `Fintech Stress Test`

    Project where I performed a stress test on the portfolio of our student managed fund [Fintech Enigma Fund](https://www.fintechenigma.no/).

    In this test I tested the portfolio under different enviornments of volatility in the market and oil-price market. 

- `EMA-SMA.ipny` & `EMA-SMA.py`

    Test enviornment and deployment code of an EMA-SMA trading algorithm that trades a portfolio of stocks. It uses mean-variance optimization to place the best hedged possition. 

- `Optios.ipynb`

    Project where i use price simulations to price options. 