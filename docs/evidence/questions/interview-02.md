# Mock interview 02: Senior Data Scientist (senior) at Swiggy

- Type / round: mixed / onsite; duration 18 min; language en; requested difficulty auto
- Company style: Case-style product analytics plus ML depth
- Skills to probe: A/B testing, Forecasting
- Candidate profile (simulated): mixed_up_down
- Interviewer LLM: ollama:qwen2.5:7b-instruct
- Checks: PASS {"questions": 16, "bank_overlap": 0, "follow_ups": 11, "follow_ups_referencing_answer": 11, "difficulty_tracks_performance": true, "difficulty_corrections_by_code": 6, "emergency_questions": 0, "fresh_questions": 0, "rejected_attempts": {"anchor_quote must be copied verbatim from the candidate's la": 1, "repeats an earlier question": 2, "follow_up must ask about what the candidate just said (the a": 1, "to return to an earlier answer use action revisit": 1}, "emergency_blueprint": false, "repeats": 0, "closed_on_time": true, "passed": true}

## Blueprint
This interview will assess the candidate's expertise in forecasting and A/B testing, as well as their ability to communicate technical solutions to non-technical stakeholders, essential for a senior data scientist role at Swiggy.

- **Forecasting Expertise** (7 min): Essential skill for developing accurate demand forecasts in a marketplace setting, as stated in the job description.
- **A/B Testing Experience** (6 min): Critical for validating hypotheses and making data-driven decisions in a dynamic marketplace, as per the job description.
- **Stakeholder Communication** (4 min): Important for bridging the gap between technical solutions and business needs, as highlighted in the job description.

## Transcript

**Interviewer** [open, c1, difficulty 3]: Hi there, I'm Maya. Let's start by discussing your experience with forecasting models. Could you walk me through a project where you developed a forecasting model for a marketplace?

**Candidate**: Hi Maya, in my previous role as a Data Scientist at [Previous Company], I led the development of a demand forecasting model for our e-commerce platform. We used Prophet and LightGBM to predict product demand for our top 500 SKUs. The model helped us optimize inventory levels, reducing stockouts by 30% and holding costs by 15%. We integrated this model into our daily reporting and decision-making processes. By leveraging historical sales data and external factors like promotions and seasonality, we were able to make more accurate forecasts. This not only improved our operational efficiency but also enhanced customer satisfaction by ensuring that products were available when customers wanted them.
  - interviewer's read: 4/5

**Interviewer** [follow_up, c1, difficulty 3]: That's great to hear. Could you elaborate on how you chose between Prophet and LightGBM? Were there any specific factors or considerations that influenced your decision?
  - builds on: "We used Prophet and LightGBM to predict product demand for our top 500 SKUs."

**Candidate**: Sure, we chose Prophet because it handled seasonality and holidays well, which was important for our business. But LightGBM was faster and gave better results in some cases, so we used both depending on the situation.
  - interviewer's read: 4/5

**Interviewer** [follow_up, c1, difficulty 3]: Could you provide some numbers to quantify the benefits of using both models in your project?
  - builds on: "Sure, we chose Prophet because it handled seasonality and holidays well, which was important for our business. But LightGBM was faster and gave better results in some cases, so we used both depending on the situation"

**Candidate**: Absolutely, in our project, we found that using both Prophet and LightGBM improved our overall forecasting accuracy. Specifically, Prophet was particularly effective for capturing seasonality and holiday effects, leading to a 25% improvement in accuracy for those periods. On the other hand, LightGBM was more efficient in handling large datasets and complex patterns, contributing to a 20% improvement in accuracy for our general demand forecasts. By combining the strengths of both models, we achieved a 35% reduction in forecast error compared to using either model alone. This dual approach not only enhanced our decision-making but also resulted in a 10% increase in inventory turnover and a 12% decrease in excess inventory, translating to significant cost savings and improved operational efficiency.
  - interviewer's read: 4/5

**Interviewer** [follow_up, c1, difficulty 4]: That's very insightful. Could you walk me through the process of how you decided which model to use for different periods? For instance, how did you determine which models were better suited for capturing seasonality versus handling complex patterns?
  - builds on: "specifically, Prophet was particularly effective for capturing seasonality and holiday effects, leading to a 25% improvement in accuracy for those periods. On the other hand, LightGBM was more efficient in handling large datasets and complex patterns, contributing to a 20% improvement in accuracy fo"

**Candidate**: We usually started with Prophet for its ease in capturing seasonality and holidays. But for more complex patterns, we used LightGBM. It worked well for large datasets. Mostly, we just tried both and saw which gave better results for each case.
  - interviewer's read: 4/5

**Interviewer** [follow_up, c1, difficulty 4]: That's interesting. Could you elaborate on the validation methods you used to ensure the accuracy and reliability of these models? How did you validate the performance of Prophet and LightGBM, and what criteria did you use to decide on their application in different scenarios?
  - builds on: "We usually started with Prophet for its ease in capturing seasonality and holidays. But for more complex patterns, we used LightGBM."

**Candidate**: Certainly, to validate the accuracy and reliability of Prophet and LightGBM, we implemented a rigorous cross-validation process. For each model, we split the historical data into training and validation sets. We used a rolling horizon approach, where we iteratively trained the model on the training set and validated it on the next time period of the validation set. This helped us assess how well the models would perform on unseen data. For Prophet, we focused on its built-in seasonality and holiday effects and used metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) to evaluate its performance. We found that Prophet performed particularly well during peak season periods, showing a 25% improvement in accuracy. For LightGBM, we used more complex evaluation metrics such as Mean Absolute Percentage Error (MAPE) and adjusted R-squared, as it was better suited for handling large datasets and complex patterns. LightGBM showed a 20% improvement in accuracy for these scenarios. Based on these metrics, we decided to use Prophet for periods with high seasonality and holidays, and LightGBM for more complex patterns. We also conducted a few A/B tests by comparing the forecasts and actual sales data to ensure the models were reliable. This dual validation approach helped us make informed decisions about which model to use in different scenarios, ultimately leading to a 35% reduction in forecast error and significant operational improvements.
  - interviewer's read: 4/5

**Interviewer** [follow_up, c1, difficulty 4]: That's great to hear. Could you tell me more about the A/B testing process you mentioned? How did you ensure the reliability of the models using this approach?
  - builds on: "and holidays, and LightGBM for more complex patterns. We also conducted a few A/B tests by comparing the forecasts and actual sales data to ensure the models were reliable"

**Candidate**: Sure, we ran about 40 A/B tests to validate the models. We would split the data into two groups, use one group to forecast with Prophet and the other with LightGBM, then compare the accuracy using metrics like MAE and RMSE. We did this multiple times to ensure the models were reliable.
  - interviewer's read: 4/5

**Interviewer** [follow_up, c1, difficulty 4]: That's great to hear. Could you provide more details on the A/B testing process you mentioned? Specifically, how did you design and implement these tests, and what were the key outcomes and learnings from these A/B tests?
  - builds on: "we ran about 40 A/B tests to validate the models"

**Candidate**: Certainly, we designed and implemented the A/B tests by splitting the historical data into two equal parts—80% for training the models and 20% for testing. For each test, we used the training data to fit both Prophet and LightGBM models. Then, we used these models to predict the demand for the test period. We compared the predicted values with the actual sales data using metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE). To ensure the reliability of the models, we conducted multiple iterations of these tests, varying the split points and the time periods. This helped us understand the model performance under different conditions. The key outcomes from these tests were that Prophet excelled in capturing seasonal and holiday effects, showing a 25% improvement in accuracy during such periods. On the other hand, LightGBM performed better for complex patterns and larger datasets, reducing forecast errors by 20%. By combining the strengths of both models, we achieved a 35% reduction in overall forecast error. The learnings from these tests were invaluable as they helped us fine-tune our model selection process. We realized the importance of using a hybrid approach, where Prophet was ideal for simpler, seasonally driven forecasts, while LightGBM was better for complex, data-intensive scenarios. This dual-model strategy significantly enhanced our forecasting accuracy and operational efficiency.
  - interviewer's read: 4/5

**Interviewer** [follow_up, c1, difficulty 4]: Could you quantify the specific performance improvements of Prophet and LightGBM in your A/B tests? For instance, what were the exact MAE and RMSE values for each model in the test period?
  - builds on: "Certainly, we designed and implemented the A/B tests by splitting the historical data into two equal parts—80% for training the models and 20% for testing."

**Candidate**: Sure, in our A/B tests, Prophet had an MAE of around 15 and an RMSE of about 20 during the test period. For LightGBM, the MAE was closer to 12 and the RMSE was around 18. Overall, the combination of both models improved our forecast accuracy, with a 35% reduction in error compared to using either model alone.
  - interviewer's read: 4/5

**Interviewer** [follow_up, c2, difficulty 4]: Thank you for that detailed explanation. Moving on to your experience with A/B testing, could you share a specific instance where you designed and implemented an A/B test to validate a hypothesis or decision within your forecasting project?
  - builds on: "an RMSE of about 20 during the test period"

**Candidate**: Certainly, one specific instance where we implemented an A/B test was to validate whether adjusting our inventory levels based on real-time sales data could improve our stock management. We split our dataset into two groups: one group where we used historical data to make inventory decisions, and another where we used real-time sales data to make those decisions using our Prophet and LightGBM models. We then compared the stockouts and holding costs between the two groups over a six-month period. The results showed that by using real-time data, we reduced stockouts by 28% and holding costs by 10%. We also observed a 15% increase in sales during periods when our inventory was well-managed. These findings supported our hypothesis and helped us implement a more dynamic inventory management strategy, which led to a 5% increase in overall profitability. This A/B test was crucial as it not only validated our approach but also provided a clear roadmap for integrating real-time data into our decision-making process.
  - interviewer's read: 4/5

**Interviewer** [follow_up, c2, difficulty 4]: That's very insightful. Could you share more details about the real-time data integration you mentioned? How did you integrate the Prophet and LightGBM models into your decision-making process, and what were the key steps in implementing this dynamic inventory management strategy?
  - builds on: "We also observed a 15% increase in sales during periods when our inventory was"

**Candidate**: Sure, we integrated the models by setting up a real-time data pipeline that fed into our decision-making tools. We used APIs to pull data from our sales system every few minutes. Then, our models would process this data to generate forecasts and recommendations. We set up alerts for when stock levels were low, so we could reorder inventory. The key steps were setting up the data pipeline, fine-tuning the models, and integrating the output into our inventory management software. This helped us make quicker, more informed decisions, reducing stockouts and holding costs.
  - interviewer's read: 4/5

**Interviewer** [new_topic, c3, difficulty 4]: How do you ensure that your technical findings are communicated clearly to non-technical stakeholders, such as business leaders or operations teams, in a way that they can understand and act upon?

**Candidate**: To ensure that my technical findings are communicated clearly to non-technical stakeholders, I use simple language and avoid jargon. I always start by summarizing the key results and their impact in layman's terms. For example, I might say, "Our demand forecasting model reduced stockouts by 30% and saved us $X in holding costs." Then, I provide visual aids like charts and graphs to illustrate the data more clearly. I also make sure to highlight the business value, such as "This improvement in forecast accuracy directly translates to a 5% increase in profitability." Additionally, I schedule regular meetings where I present the findings and walk through the data step-by-step. I encourage feedback and questions to ensure that everyone understands the implications and can act on the insights. By breaking down the technical details into actionable points, I help stakeholders make informed decisions and drive better business outcomes.
  - interviewer's read: 4/5

**Interviewer** [follow_up, c3, difficulty 4]: That's very helpful. Could you give an example of a time when you faced a challenge in communicating your findings to non-technical stakeholders, and how you overcame it?
  - builds on: "Then, I provide visual aids like charts and graphs to illustrate the data more clearly."

**Candidate**: Sure, once we launched a new inventory management system, and the IT team had a hard time understanding how the Prophet and LightGBM models worked. They were concerned about the complexity and wanted to know how it would affect their daily operations. To address this, I created a simple explanation and a demo. I used basic terms and analogies, like comparing the models to traffic lights that help manage inventory levels. I also set up a training session with the IT team, where I walked them through the system step-by-step. This helped them see how the models worked and how they could use the system to their advantage. By being patient and explaining things clearly, we were able to get everyone on board and use the system effectively.
  - interviewer's read: 4/5

**Interviewer** [new_topic, c2, difficulty 4]: Could you share more details about the design and statistical significance of the A/B test you implemented to validate the hypothesis about the inventory management system?
  - builds on: "we were able to get everyone on board"

**Candidate**: Certainly, during the A/B test to validate the hypothesis about the inventory management system, we split our dataset into two equal parts. The first group used historical data to make inventory decisions, while the second group used real-time data from the new Prophet and LightGBM models. To ensure statistical significance, we used a significance level of 0.05 (95% confidence level). We ran the test over a six-month period, collecting daily sales data and comparing stockout rates and holding costs between the two groups. We used statistical tests, such as the Chi-squared test, to determine if the differences in stockouts and holding costs were significant. The results showed that the group using real-time data had a 28% reduction in stockouts and a 10% reduction in holding costs compared to the group using historical data. These results were statistically significant, as the p-value was below our threshold of 0.05. The test not only validated our hypothesis but also provided clear evidence to support the implementation of the new system, leading to a 5% increase in overall profitability.
  - interviewer's read: 4/5

**Interviewer** [follow_up, c2, difficulty 4]: That's very insightful. Could you explain more about the statistical methods you used to validate the significance of the differences in stockouts and holding costs between the two groups?
  - builds on: "the group using real-time data had a 28% reduction in stockouts and a 10% reduction in holding costs compared to the group using historical data."

**Candidate**: Sure, we used the Chi-squared test to validate the significance of the differences. We compared the observed stockout rates and holding costs against the expected rates based on our models. The Chi-squared test helped us determine if the differences were statistically significant at a 95% confidence level. The p-value from the test was below 0.05, confirming that the improvements in stockouts and holding costs were not due to chance.
  - interviewer's read: 4/5

**Interviewer** [wrap_up, c2, difficulty 4]: That's very insightful. Moving on to your experience with stakeholder communication, could you provide an example of a time when you had to present complex technical findings to a non-technical audience, and how you ensured they understood and acted on the insights?
  - builds on: "We used the Chi-squared test to validate the significance of the differences. The p-value from the test was below 0.05, confirming that the improvements in stockouts and holding costs were not due to chance."

**Candidate**: Certainly, one example was when I presented the findings of our demand forecasting model to the executive team. Initially, they were concerned about the complexity of the models and how it would affect their decision-making. To ensure they understood, I prepared a simplified presentation that included: 1. **Key Metrics**: I focused on key metrics like MAE and RMSE, explaining them in simple terms. For instance, I said, "Our model reduced forecast errors by 30%," and provided visual aids like charts showing the pre- and post-model forecast accuracy. 2. **Business Impact**: I highlighted the business benefits in layman's terms. For example, I said, "This improvement directly translates to a 10% reduction in holding costs and a 5% increase in sales." 3. **Real-Life Examples**: I used real-life examples. For instance, I showed a scenario where a product was in high demand during a holiday season and how our model predicted and managed the stock levels to avoid stockouts. 4. **Interactive Q&A**: I scheduled a Q&A session where I answered questions and clarified any doubts. This helped everyone understand the implications and feel comfortable with the model. 5. **Actionable Steps**: I provided clear, actionable steps they could take, such as integrating the model into their daily reporting and decision-making processes. By breaking down the technical details and focusing on the business value, the executive team was able to see the practical benefits and felt confident in adopting the model.
  - interviewer's read: 4/5

**Interviewer** [close, c3, difficulty 4]: Thank you for sharing that example. It was clear and practical. You have a strong track record in both forecasting and A/B testing, and you've effectively communicated your findings to non-technical stakeholders. It was a pleasure interviewing you. Goodbye.
  - builds on: "This improvement directly translates to a 10% reduction in holding costs and a 5% increase in sales."
