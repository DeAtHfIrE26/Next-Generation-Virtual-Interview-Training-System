# Mock interview 12: Data Analyst (intern) at Zomato

- Type / round: technical / screening; duration 10 min; language en; requested difficulty auto
- Company style: SQL and business sense, friendly
- Skills to probe: SQL, Excel
- Candidate profile (simulated): average
- Interviewer LLM: ollama:qwen2.5:7b-instruct
- Checks: PASS {"questions": 12, "bank_overlap": 0, "follow_ups": 7, "follow_ups_referencing_answer": 7, "difficulty_tracks_performance": true, "difficulty_corrections_by_code": 3, "emergency_questions": 0, "fresh_questions": 0, "rejected_attempts": {"anchor_quote must be copied verbatim from the candidate's la": 3, "repeats an earlier question": 2}, "emergency_blueprint": false, "repeats": 0, "closed_on_time": true, "passed": true}

## Blueprint
This screening interview assesses the candidate’s SQL and Excel skills, as well as their ability to demonstrate business acumen, which are crucial for a Data Analyst role at Zomato.

- **SQL** (4 min): Data Analysts need strong SQL skills to manipulate and analyze large datasets.
- **Excel** (3 min): Proficiency in Excel is essential for Data Analysts to perform complex data analysis and visualization.
- **Business Acumen** (3 min): Demonstrating an understanding of how data analysis can drive business decisions is critical for this role.
- **Problem-Solving** (0 min): Proving the ability to tackle complex problems using data analysis is important for the role.

## Transcript

**Interviewer** [open, c1, difficulty 3]: Hi there! I'm Maya. Let's start by discussing your SQL skills. Can you walk me through a specific project where you used SQL to retrieve or manipulate data?

**Candidate**: Sure, Maya. I worked on a project where we needed to analyze customer feedback on restaurant services. I used SQL to extract feedback data from our CRM system, filtering out any irrelevant entries. Then, I cleaned the data to get a clear view of the positive and negative comments. With SQL queries, I grouped the feedback by restaurant and month, which helped us identify trends over time. This allowed us to focus on areas needing improvement and measure the effectiveness of our changes.
  - interviewer's read: 4/5

**Interviewer** [follow_up, c1, difficulty 4]: That's great! Could you tell me more about the tools or databases you used in this project?
  - builds on: "Sure, Maya. I worked on a project where we needed to analyze customer feedback on restaurant services."

**Candidate**: Certainly! For this project, I used MySQL as our main database. I also leveraged Python for some data manipulation tasks, particularly when I needed to join data from different sources. I found that combining SQL with Python made it easier to handle large datasets and perform complex queries more efficiently.
  - interviewer's read: 4/5

**Interviewer** [follow_up, c1, difficulty 4]: That's great! You mentioned using MySQL. Could you elaborate on why you chose MySQL for this project and how it fit into the overall data analysis process?
  - builds on: "I used MySQL as our main database"

**Candidate**: Sure, Maya. I chose MySQL because it's robust and widely used, making it easy to get support and resources. It also supports complex queries, which was crucial for filtering and cleaning the data. For the project, I used MySQL to store and manage the customer feedback data, then wrote SQL queries to extract the relevant information. This fit well into the data analysis process as it allowed us to efficiently handle and manipulate the data, ensuring we could quickly identify trends and areas for improvement.
  - interviewer's read: 4/5

**Interviewer** [follow_up, c1, difficulty 4]: That's great. Could you share some specific SQL queries you used in this project and how they contributed to the analysis and insights?
  - builds on: "I chose MySQL because it's robust and widely used, making it easy to get support and resources."

**Candidate**: Sure, Maya. For this project, I used a few key SQL queries. One of them was to extract feedback data by restaurant and month: ```sql SELECT restaurant_id, DATE_FORMAT(feedback_date, '%Y-%m') AS month, COUNT(*) AS feedback_count FROM customer_feedback GROUP BY restaurant_id, month ORDER BY restaurant_id, month; ``` This query helped us see the number of feedback entries for each restaurant over time. Another query I used was to filter out irrelevant entries: ```sql SELECT * FROM customer_feedback WHERE feedback_date BETWEEN '2022-01-01' AND '2023-01-01' AND feedback_text LIKE '%service%'; ``` This one focused on feedback containing the word "service" within a specific time frame. These queries were instrumental in identifying trends and understanding customer opinions, which in turn helped us improve our service quality.
  - interviewer's read: 4/5

**Interviewer** [follow_up, c1, difficulty 4]: Great! Can you walk me through the optimization techniques you used to ensure the SQL queries ran efficiently on large datasets?
  - builds on: "Sure, Maya. For this project, I used a few key SQL queries."

**Candidate**: Certainly, Maya. To optimize the SQL queries, I made sure to create indexes on the columns that were frequently used in the WHERE and JOIN clauses. For example, I created an index on the `restaurant_id` and `feedback_date` columns in the `customer_feedback` table. This helped speed up the filtering and grouping processes. Additionally, I used appropriate data types and ensured that the data was normalized to reduce redundancy and improve query performance. I also made sure to limit the data being processed by adding specific date ranges and filtering out unnecessary data early in the query. Finally, I tested the queries on a smaller subset of data first to identify any performance bottlenecks. Once I found the issues, I adjusted the queries and retested until they ran smoothly and efficiently on larger datasets.
  - interviewer's read: 4/5

**Interviewer** [follow_up, c1, difficulty 4]: That's excellent! You mentioned creating indexes to optimize the queries. Could you provide an example of how you determined which columns to index and the impact it had on query performance?
  - builds on: "Maya. To optimize the SQL queries, I made sure to create indexes on the columns that were frequently used in the WHERE and JOIN clauses. For example, I created an index on the `restaurant_id` and `feedback_date` columns in"

**Candidate**: Sure, Maya. In the project, I identified that the `restaurant_id` and `feedback_date` columns were frequently used in the WHERE and JOIN clauses. By creating indexes on these columns, I was able to significantly reduce the query execution time. For instance, the `restaurant_id` column was used to group feedback by restaurant, and the `feedback_date` column was used to filter feedback within specific time frames. This reduced the number of rows that needed to be scanned, making the queries much faster. The impact was noticeable; a query that previously took several minutes to execute could now run in just seconds. This improvement was crucial for real-time analysis and reporting, as it allowed us to get insights quickly and make timely decisions.
  - interviewer's read: 4/5

**Interviewer** [follow_up, c1, difficulty 4]: That's great! Could you provide some specific metrics on the performance improvements you observed after applying these optimizations?
  - builds on: "This improvement was crucial for real-time analysis"

**Candidate**: Certainly, Maya. After applying the optimizations, we observed a significant improvement in query performance. For example, a query that originally took about 30 seconds to run was reduced to just 3 seconds. This not only sped up our data analysis but also allowed us to run more complex queries in real-time. We saw a 95% reduction in query execution time, which was a substantial improvement. This efficiency gain was particularly useful for our real-time reporting dashboard, where quick data retrieval is crucial for making informed decisions.
  - interviewer's read: 4/5

**Interviewer** [follow_up, c1, difficulty 4]: Could you give me an example of a particularly challenging query you had to optimize? How did you approach it and what was the outcome?
  - builds on: "After applying the optimizations, we observed a significant improvement in query performance."

**Candidate**: Certainly, Maya. One particularly challenging query involved aggregating feedback data across multiple restaurants and calculating sentiment scores for each period. The initial query was quite complex and slow, running for several minutes. To optimize it, I first identified the bottlenecks by using EXPLAIN to analyze the query execution plan. I noticed that the query was performing full table scans and joining multiple large tables, which were time-consuming. To address this, I created indexes on the join columns and used more specific date ranges to limit the data. Additionally, I refactored the query to use temporary tables to store intermediate results, reducing the load on the main query. After these changes, the query's execution time dropped from several minutes to just a few seconds. This optimization not only improved the performance but also allowed us to run the query more frequently for real-time insights. The outcome was a much faster and more efficient data analysis process, enabling quicker decision-making.
  - interviewer's read: 4/5

**Interviewer** [new_topic, c2, difficulty 4]: That's great! Moving on to Excel, can you give me an example of how you used pivot tables, formulas, or macros to analyze or visualize data in a project?

**Candidate**: Certainly, Maya. In a project where we needed to analyze sales data across different regions and time periods, I used pivot tables in Excel to summarize and visualize the data. I created a pivot table to aggregate sales figures by region and month, which helped us identify which regions were performing well and which needed more attention. For further analysis, I used formulas like VLOOKUP and INDEX-MATCH to cross-reference sales data with customer demographics, allowing us to see if there were any correlations between customer age groups and purchasing behavior. Additionally, I created macros to automate repetitive tasks, such as updating the pivot table with new data and generating reports. This automation saved a lot of time and helped us focus on more strategic analysis. Overall, Excel was a powerful tool that helped us make sense of large datasets and draw meaningful insights quickly.
  - interviewer's read: 4/5

**Interviewer** [new_topic, c3, difficulty 4]: That's great! Let's talk about how data analysis can drive business decisions. Can you give me an example of how you connected data insights to specific business objectives in a project?

**Candidate**: Sure, Maya. In one project, we were looking to improve customer satisfaction at our partner restaurants. I analyzed customer feedback data to identify common issues such as long wait times and poor service. By using SQL to group feedback by restaurant and month, I was able to pinpoint which restaurants had the most complaints. This data was then visualized in a dashboard using Excel, showing trends over time and highlighting the areas needing improvement. We shared these insights with the restaurant managers and provided them with specific action items, like increasing staff during peak hours or providing better training for service staff. As a result, the feedback complaints for the targeted restaurants decreased by 30% within three months. This improvement not only enhanced the customer experience but also helped the restaurants meet their business objectives of higher customer satisfaction and better revenue.
  - interviewer's read: 4/5

**Interviewer** [wrap_up, c3, difficulty 4]: That's a great example! How do you feel about your overall skills in data analysis and how they can drive business decisions?
  - builds on: "This improvement not only enhanced the customer experience but also helped the restaurants meet their business objectives of higher customer satisfaction and better revenue."

**Candidate**: I feel really confident in my data analysis skills and how they can drive business decisions, Maya. I believe that by leveraging tools like SQL and Excel, I can extract meaningful insights from data that help guide strategic decisions. Whether it's optimizing customer feedback to improve service quality or analyzing sales data to make informed business choices, I'm always looking for ways to turn data into actionable steps. I think my ability to clean and manipulate data, combined with my experience in visualization and reporting, makes me well-equipped to contribute to data-driven decisions. I'm excited about the opportunity to bring these skills to Zomato and help drive growth and improvement in the business.
  - interviewer's read: 4/5

**Interviewer** [close, c3, difficulty 4]: Thank you for sharing your insights. It was great discussing your skills and experience. We're excited to have you join us at Zomato!
  - builds on: "I feel really confident in my data analysis skills and how they can drive business decisions, Maya."
