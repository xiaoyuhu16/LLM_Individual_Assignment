# 94812-A3_Individual_Assignment

## Stock Price Prediction and GameStop Short Squeeze
**Background**

In late January 2021, GameStop (GME), a video game retailer, became the center of a financial phenomenon known as a 'short squeeze.' This occurred when a surge of retail investors, coordinating through social media platforms like Reddit's r/wallstreetbets, began buying up GameStop's stock. This drove up the stock price dramatically, which in turn inflicted heavy losses on hedge funds and other investors who had bet against the stock by short-selling it. The event drew widespread media attention, sparked controversy over stock market practices, and led to hearings in the U.S. Congress.

**Objective**

To build a stock price prediction model incorporating both historical data and social media sentiment, evaluate its accuracy on the GameStop short squeeze, and analyze potential improvements based on the event.

Table of Contents
1. Model Building

   i. Data acquisition
   
   ii. Feature Engineering
   
   iii. Model Building
  
2. Retrospective Predictions Evaluation
   
   i. Prediction Period
   
   ii. Evaluation
   
   iii. Visualization
   
3. GameStop Short Squeeze and Model Adaptation
   
   i. Event Analysis
   
   ii. Model Sensitivity
   
   iii. Algorithmic Adjustments
   
4. Conclusion and Future Directions
   
   i. Summarize
   
   ii. Discuss
   
   iii. Propose


## Part I. Model Building
**(i) Data acquisition**

     Han, Jing, 2022, "Reddit Dataset on Meme Stock: GameStop", https://doi.org/10.7910/DVN/TUMIPC, Harvard Dataverse, V3, UNF:6:c9s1zhZLHH+k32UmoPZu7A== [fileUNF]
     
     Gabriel Preda. (2021). Reddit WallStreetBets Posts [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/2530155

     yfinance

**(ii) Feature Engineering**

Time Series Forecasting on Stock History Data:

- Standardize the dataset using MaxAbsScaler()

- Create sequences of length 13 after trials. The length yields the lowest MSE, RMSE, and MAE.

Sentiment Analysis on Reddit Posts Datasets:

- Combine post titles and body of posts for more accurate prediction of the post sentiment

- Design sentiment score based on VADER, where positive sentiment is 1, neutral sentiment is 2, and negative sentiment is 3


**(iii) Model Building**
Time Series Forecasting on Stock History Data:

- LSTM model with 4 layers and one Dense layer
- Adam optimizer
- Mean_Squared_Error

Sentiment Analysis on Reddit Posts Datasets:
- VADER

Model fusion:
- Combine sentiment datasets:
  1. Reddit WallStreetBets Posts with sentiment score calculated with VADER
  2. Reddit Dataset on Meme Stock: GameStop with sentiment score given(using VADER as well)
- Feature Concatenation
  Merge sentiment scores in Reddit posts with Stock Price history data
- Standardize combined datasets using MaxAbsScalar() as well
- Create sequences of length 13 after trials. The length yields the lowest MSE, RMSE, and MAE
- The same LSTM model with 4 layers and one Dense layer
- Adam optimizer
- Mean_Squared_Error

## Part II. Retrospective Predictions and Evaluation

**Prediction Period**

- From 2021-06-01 To 2021-08-31

**Evaluation**

***Before Adding Sentiment Analysis:***

<img width="306" alt="image" src="https://github.com/xiaoyuhu16/LLM_Individual_Assignment/assets/108830414/7185507c-b30c-46d1-86cd-445804f249f5">

- Mean Squared Error (MSE): 80.37133160328221
  
- Root Mean Squared Error (RMSE): 8.965005945524085
  
- Mean Absolute Error (MAE): 6.88510000705719

***After Adding Sentiment Analysis:***

<img width="310" alt="image" src="https://github.com/xiaoyuhu16/LLM_Individual_Assignment/assets/108830414/4d40dd8c-054b-4de9-a334-0f9edbce219d">

- Mean Squared Error (MSE): 165.8012483058564

- Root Mean Squared Error (RMSE): 12.876383355036321

- Mean Absolute Error (MAE): 9.544308254941228

**Visualization**

- Before Adding Sentiment Analysis:

<img width="985" alt="image" src="https://github.com/xiaoyuhu16/LLM_Individual_Assignment/assets/108830414/c917f59f-efc5-4497-b308-a9457da797d7">


- After Adding Sentiment Analysis:

<img width="981" alt="image" src="https://github.com/xiaoyuhu16/LLM_Individual_Assignment/assets/108830414/7336b205-e8c3-4171-a2d6-20354a0455fb">


## Part III. GameStop Short Squeeze and Model Adaptation

**Event Analysis**

<div style="display:flex;">
  <img alt="image" src="https://github.com/xiaoyuhu16/LLM_Individual_Assignment/assets/108830414/77f40258-e792-45d8-b280-d8a444123eca" width="50%"><img alt="image" src="https://github.com/xiaoyuhu16/LLM_Individual_Assignment/assets/108830414/15986c6c-a952-409b-a221-e03be80f96a8" width="50%">
</div>

The two plots above show the Average Sentiment Score Per Day and Sentiment Scores Stacked Bar Chart from Jan.2021 to Aug.2021, covering the complete period studied in this project(training + testing).

From the plots, we can conclude that the average sentiment expressed in Reddit Posts was less positive in the first half of the year than in the second half. In the left plot, the average sentiment score per day shows a subtle increase in score and decreases gradually after February. Recall that the score scale is 3 for negative, 2 for neutral, and 1 for positive, so we can conclude that the posts on Reddit started to express relatively less positive sentiment since late January/early February. However, the right plot shows a more interesting trend. There were more than 600,000 posts in late February, which is significantly greater in scale than all the other periods. It is so abnormal that we can conclude GameStop is being hotly and intentionally discussed during that time.

**Model Sensitivity**

***- Sensitivity 1: Change sentiment of 2021-05-13 to 2021-05-28 to all positive***

<img width="982" alt="image" src="https://github.com/xiaoyuhu16/LLM_Individual_Assignment/assets/108830414/f59006f0-ae07-4450-af47-5106e8daf241">

By modifying the sentiment of the posts from the last 12 days in the training dataset, the predicted closing price for the testing dataset(starting at 2021-06-01) is generally higher than the ones calculated with actual sentiment scores from the 12 days. This could be evidence that a short period of positive sentiments would potentially lead to an increase in the predicted stock closing price.

***- Sensitivity 2: Change sentiment of 2021-05-13 to 2021-05-28 to all negative***

<img width="993" alt="image" src="https://github.com/xiaoyuhu16/LLM_Individual_Assignment/assets/108830414/9a09f160-794e-40c8-9027-24c3638ad8a2">

By modifying the sentiment of the posts from the last 12 days in the training dataset, the predicted closing price for the testing dataset(starting at 2021-06-01) drops dramatically in early June. One possible explanation is that the negative sentiments posted in late May significantly influenced the customers' behavior and caused a dramatic decrease.


***- Sensitivity 3: Change sentiment of 2021-05-13 to 2021-05-27 to all positive and 2021-05-28 to negative***

<img width="979" alt="image" src="https://github.com/xiaoyuhu16/LLM_Individual_Assignment/assets/108830414/423bfef3-073b-427e-b750-7d7e9bd59ddb">

Adding a sudden change in sentiment, from positive to negative, to the last day in the training dataset has the same effect on the predicted closing price for the testing dataset as in Sensitivity 2. This shows that the model is very sensitive to sudden changes in sentiment.  

***-Sensitivity 4: Change sentiment of 2021-05-13 to 2021-05-27 to all negative and 2021-05-28 to positive***

<img width="974" alt="image" src="https://github.com/xiaoyuhu16/LLM_Individual_Assignment/assets/108830414/22859007-3f77-48c4-866d-d48b717ea5ee">

Similar to the case in sensitivity test 3, with only a single day of positive sentiment on the last day switching from a period of negative sentiment posts is the same as having consistent positive sentiment for a longer period. The model seems to depend on the sentiment score from the most recent day to predict future closing prices.

***-Sensitivity 5: Change sentiment of 2021-05-1 to 2021-05-27 to all negative and 2021-05-28 to positive***

<img width="1001" alt="image" src="https://github.com/xiaoyuhu16/LLM_Individual_Assignment/assets/108830414/a1920e73-6c85-4a2a-9ad3-c45a400ee647">

Modifying the sentiment score for days far away(longer than the sequence_length variable) from the predicted dates does not have any influence on the prediction. Therefore, the model is insensitive to changes made before the date used in time series forecasting.


**Algorithmic Adjustments**
1. Using MinMaxScaler()
   
   Instead of MaxAbsScaler(), use MinMaxScaler to standardize input data. MinMaxScaler is generally more sensitive to outliers than the MaxAbsScaler. This sensitivity arises because the MinMaxScaler rescales the data based on the minimum and maximum values of the feature, which can be heavily influenced by extreme outlier values. However, we may need sensitivity to capture the extreme social media sentiment.

2. Use different sentiment analysis Techniques
   
   Instead of using VADER, use BERT. Because many of the posts' contexts are constituted of only titles, which are usually short, BERT might be better at capturing the sentiment in such a short phrase as it uses a bidirectional transformer encoder.

3. For Reddit Dataset on Meme Stock: GameStop Dataset, use the variable 'compound', or 'pos', 'neu', and 'neg' in combination to determine the sentiment of the posts

    Distribution of sentiment using VADER score in the dataset: 957936 Positive, 21955 Neutral, and 6184 Negative.
  
    Distribution of sentiment determined by 'compound': 401653 Positive, 438972 Neutral, and 192611 Negative.
  
    Distribution of sentiment determined by 'pos', 'neu', and 'neg': 957936 Positive, 915162 Neutral, and 34453 Negative.

4. More complex model fusion approach

   Instead of fusing the two models by feature concatenation, process the sentiment score through another linear layer in LSTM. In that case, the model will understand the features more thoroughly.

5. More features related to Financial news added

   Incorporate information such as the number of posts each day into the LSTM model since the number of posts is also related to the ShortSqueeze Event.
   

## Part IV. Conclusion and Future Directions

**Summarize**

Although the MSE, RMSE, and MAE did not improve for the LSTM model with sentiment scores added, the predicted line of closing prices showed a more obvious trend that matched the distribution of actual closing prices. 

<div style="display:flex;">
<img alt="image" src="https://github.com/xiaoyuhu16/LLM_Individual_Assignment/assets/108830414/c917f59f-efc5-4497-b308-a9457da797d7" width="50%"><img alt="image" src="https://github.com/xiaoyuhu16/LLM_Individual_Assignment/assets/108830414/7336b205-e8c3-4171-a2d6-20354a0455fb" width="50%">
</div>

Combined results from sensitivity analysis, we can conclude that posts' sentiment from the most recent dates will have a dramatic influence on the predicted stocking price. The model with sentiment scores added is better at capturing the increase and decrease in the stock price data, and sudden changes in sentiment score for dates close to the predicted period will significantly change the trend of the stock price.

However, the model has several limitations.
- The data used for sentiment analysis contains only posts from Reddit.
- Sentiment analysis is performed mostly on only post titles, which are usually short and may not make sense.
- Model fusion is based on only feature concatenation, which only takes in the average sentiment score from each day without considering other factors like the number of posts and the magnitude of sentiment.

**Discuss**

From the result, the GameStop short squeeze has a moderate impact on the effectiveness of traditional forecasting models. When incorporating media sentiment data into traditional forecasting models, the model becomes better at capturing the flow of the event and stock price fluctuations. However, this also results in greater MSE, RMSE, and MAE when predicting closing prices. 

The ethics of social media mining for financial analysis, like the study conducted on GameStop's stock prediction using LSTM models and sentiment analysis from Reddit, involves navigating the intersection of privacy, transparency, and the implications of predictive analytics. While public posts on social media platforms are accessible for analysis, the ethical use of this data requires consideration of individual privacy and informed consent. The interpretation of such data must be handled with care, acknowledging the limitations of sentiment analysis tools, which may not always capture the nuances of human communication. Additionally, the potential influence of aggregated sentiment on market behaviors prompts ethical questions about market fairness and manipulation. There is a need for ethical frameworks that guide the responsible use of public data, ensuring that analyses do not misrepresent or disproportionately impact certain groups or individuals. This research reflects the broader challenges and responsibilities inherent in integrating social media data into predictive financial models. Future studies could build upon this work by exploring more diverse data sources, improving representativeness, and developing ethical guidelines to govern the use of public data in financial forecasting.

**Propose**

In light of the findings from the current study, future research directions could be multidimensional, aiming at refining the accuracy and applicability of stock price prediction models that incorporate social media sentiment. Here are several proposals:

1. Data Enrichment: Expand the sentiment analysis to include a wider range of social media platforms beyond Reddit, such as Twitter, StockTwits, and financial forums. This will provide a more comprehensive sentiment landscape, potentially improving the model's predictive capabilities by capturing a broader spectrum of investor sentiment.

2. Deep Sentiment Analysis: Instead of focusing on post titles, implement deep learning models capable of understanding context, sarcasm, and complex sentiment expressions in longer text bodies. Natural Language Processing (NLP) advancements, such as transformer-based models like BERT or GPT-3, could be used to analyze entire posts and threads for a more nuanced sentiment understanding.

3. Hybrid Models: Develop hybrid models that combine sentiment analysis with other data sources like financial news, company earnings reports, and macroeconomic indicators. These models could use ensemble techniques or multi-input neural networks to process different data streams.

4. Sentiment Magnitude and Post Volume: Consider not only the average sentiment score but also the volume of posts and the magnitude of expressed sentiments. High volumes of strongly negative or positive posts could have a more pronounced impact on stock prices.

5. Ethical Frameworks: Establish clear ethical frameworks for social media data mining, addressing privacy concerns and ensuring transparency about data usage. This involves the development of models that are robust to manipulation, such as artificially generated sentiment through coordinated campaigns or bots.

6. Backtesting and Re-testing: Employ rigorous backtesting strategies on historical data to simulate the model's performance in different market conditions, helping to fine-tune it and assess its robustness and reliability over time.



## Appendix

Tools Used: ChatGPT-4
