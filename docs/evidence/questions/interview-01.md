# Mock interview 01: Software Engineer (junior) at Infosys

- Type / round: technical / screening; duration 12 min; language en; requested difficulty auto
- Company style: Fundamentals, OOP, DBMS and a project walkthrough
- Skills to probe: Data structures, SQL
- Candidate profile (simulated): weak
- Interviewer LLM: ollama:qwen2.5:7b-instruct
- Checks: PASS {"questions": 16, "bank_overlap": 0, "follow_ups": 10, "follow_ups_referencing_answer": 10, "difficulty_tracks_performance": true, "difficulty_corrections_by_code": 0, "emergency_questions": 0, "fresh_questions": 0, "rejected_attempts": {"repeats an earlier question": 4, "close ends the interview, so it must not ask anything": 1}, "emergency_blueprint": false, "repeats": 0, "closed_on_time": true, "passed": true}

## Blueprint
This screening interview will assess the candidate's foundational knowledge in data structures and SQL, as well as their ability to take ownership of projects and demonstrate measurable impact, aligning with the role's requirements.

- **Data Structures** (4 min): Critical for efficient coding and problem-solving, as highlighted in the job description.
- **SQL** (3 min): Important for database management and integration, as shown in the candidate's projects.
- **Project Ownership and Impact** (3 min): Demonstrating how the candidate takes ownership of projects and delivers measurable impact, as seen in their internship.
- **Problem-Solving and Code Efficiency** (2 min): Relevant for a junior software engineer role, leveraging the project experiences mentioned.

## Transcript

**Interviewer** [open, c1, difficulty 3]: Hi there, I'm Maya. Let's start with data structures. Can you explain what a stack and a queue are and how they are typically used in software development?

**Candidate**: Stack and queue, right. Stack, it's like a plate stack where you add and remove plates from the top. Queue, it's like a line where you join at the back and leave from the front. Used for like, stack for undo, queue for printing tasks.
  - interviewer's read: 3/5

**Interviewer** [follow_up, c1, difficulty 3]: That's a good start. Can you now compare a stack and a queue in terms of their characteristics and how these differences can impact their use in software development, especially in the context of the role you're applying for at Infosys?
  - builds on: "Stack, it's like a plate stack where you add and remove plates from the top. Queue, it's like a line where you join at the back and leave from the front. Used for like, stack for undo, queue for printing tasks."

**Candidate**: Stack, it's LIFO, like plates. Queue, FIFO, line at the front. Stack, for undo things, queue, for tasks. Differences, stack faster for top, queue for order. Infosys, maybe for task order, stack for undo functions.
  - interviewer's read: 3/5

**Interviewer** [follow_up, c1, difficulty 3]: Got it. Can you elaborate on the characteristics of a stack and a queue, such as time complexity for common operations, and how these characteristics influence their use in software development?
  - builds on: "Stack, it's LIFO, like plates. Queue, FIFO, line at the front."

**Candidate**: Stack, LIFO, push and pop are fast, O(1). Queue, FIFO, enqueue and dequeue, also fast, O(1). For Infosys, maybe stack for undo, queue for task order, right?
  - interviewer's read: 3/5

**Interviewer** [follow_up, c1, difficulty 3]: Can you provide an example of a scenario in your project or internship where you had to use a stack or a queue, and how understanding their characteristics, like time complexity, influenced your decision-making or solution design?
  - builds on: "Stack, LIFO, push and pop are fast, O(1). Queue, FIFO, enqueue and dequeue, also fast, O(1)."

**Candidate**: In my internship, I used a stack for the Flask REST API to handle request history. For time complexity, push and pop being O(1) was good for managing requests. It helped with implementing the undo feature, making the API more user-friendly.
  - interviewer's read: 3/5

**Interviewer** [follow_up, c1, difficulty 3]: That's a great example. Could you walk me through how understanding the time complexity of stack operations, such as push and pop being O(1), influenced your design choices or problem-solving approach in the project?
  - builds on: "Stack, for the Flask REST API to handle request history. For time complexity, push and pop being O(1) was good for managing requests."

**Candidate**: Sure, understanding that push and pop are O(1) helped in designing the API to handle requests efficiently. It made sure that adding or removing requests was quick, which was good for performance. So, I focused on using stack for managing request history, making it easy to implement the undo feature without slowing down the API.
  - interviewer's read: 3/5

**Interviewer** [follow_up, c1, difficulty 3]: That's a good example. Could you elaborate on how the understanding of stack operations specifically influenced the design and optimization of the API, especially in terms of performance and user experience?
  - builds on: "So, I focused on using stack for managing request history, making it easy to implement the undo feature without slowing down the API."

**Candidate**: Sure, understanding that push and pop operations are O(1) helped in optimizing the API's request handling. It made sure that adding or removing requests was quick, improving performance. For users, it meant faster response times and smoother interactions, especially when using the undo feature.
  - interviewer's read: 3/5

**Interviewer** [follow_up, c1, difficulty 3]: Could you give a specific number or metric that demonstrates how using a stack improved the performance of your API?
  - builds on: "It made sure that adding or removing requests was quick, improving performance."

**Candidate**: Sure, using a stack helped reduce the response time for request handling by about 20% compared to using a simple list. It made the API more responsive and smoother for users.
  - interviewer's read: 4/5

**Interviewer** [challenge, c1, difficulty 3]: That's a great example. Could you walk me through the steps you took to implement the stack in your Flask API and how you verified that the 20% improvement in response time was indeed due to the stack implementation? Also, how did you ensure that the stack was the right choice for this particular use case?
  - builds on: "using a stack helped reduce the response time for request handling by about 20% compared to using a simple list."

**Candidate**: Sure, for the stack implementation, I first defined a class with push and pop methods. Then, in the API, I used this stack to manage request history. To verify the 20% improvement, I used a timer before and after adding and removing requests. The stack made these operations faster, leading to quicker response times. For the right choice, I compared it with a list and saw the stack was more efficient for undo functionality.
  - interviewer's read: 4/5

**Interviewer** [new_topic, c2, difficulty 3]: Great example. Now, let's move on to SQL. Can you explain how you would optimize a query to improve the performance of a database in one of your projects?
  - builds on: "Sure, for the stack implementation, I first defined a class with push and pop methods. Then, in the API, I used this stack to manage request history."

**Candidate**: For the chat app, we had a lot of messages, and sometimes the queries were slow. I added indexes to the timestamp column to speed things up. Also, used LIMIT to control the number of messages returned. Think it helped a bit, but not sure by how much.
  - interviewer's read: 3/5

**Interviewer** [follow_up, c2, difficulty 3]: Can you provide more details on how you added the indexes and the LIMIT clause to the query? Also, did you measure the performance improvement, and if so, could you share any specific metrics or benchmarks you used?
  - builds on: "I added indexes to the timestamp column to speed things up. Also, used LIMIT to control the number of messages returned."

**Candidate**: Sure, for the chat app, I added indexes to the timestamp column to speed up the query when fetching recent messages. Also, used LIMIT to control how many messages were returned. I think it helped, but didn't measure it exactly. Maybe it was faster, but not sure by how much.
  - interviewer's read: 3/5

**Interviewer** [follow_up, c2, difficulty 3]: Can you provide specific performance metrics or benchmarks that demonstrate the improvement you mentioned? Also, how did you ensure that adding indexes and using LIMIT was the right choice for optimizing the query?
  - builds on: "I think it helped, but didn't measure it exactly. Maybe it was faster, but not sure by how much."

**Candidate**: Sure, for the chat app, I added indexes to the timestamp column and used LIMIT to control the number of messages. I noticed that the response time for fetching messages improved, but I didn't have exact metrics. I compared it with a query without indexes and LIMIT, and it felt faster. I think it was a good choice, but I didn't measure the exact improvement.
  - interviewer's read: 3/5

**Interviewer** [revisit, c3, difficulty 3]: Can you share any specific performance metrics or benchmarks that demonstrate the improvement you mentioned with the chat app project?
  - builds on: "Sure, for the chat app, I added indexes to the timestamp column and used LIMIT to control the number of messages."

**Candidate**: Sure, I noticed that the response time for fetching messages improved, but I didn't measure it exactly. I compared it with a query without indexes and LIMIT and it felt faster. I think it was a good choice, but I didn't have exact metrics to show the improvement.
  - interviewer's read: 3/5

**Interviewer** [new_topic, c4, difficulty 3]: Can you describe your approach to solving a problem that required efficient coding, and how you ensured your solution was optimized for performance?

**Candidate**: Sure, in the chat app project, I faced a problem where the server was slowing down due to handling too many WebSocket connections. I started by identifying the bottleneck, which was the processing of messages. I optimized the code by reducing unnecessary computations and using more efficient data structures like the stack for managing message history. I also made sure to use appropriate indexing and LIMIT in SQL queries to reduce database load. To ensure the solution was optimized, I tested the code with different loads and compared it with the previous version. It helped a lot, but I didn't have exact metrics to show the improvement.
  - interviewer's read: 3/5

**Interviewer** [new_topic, c3, difficulty 3]: Can you describe how you ensured your work on the chat app project was well-organized and contributed to the overall success of the project?
  - builds on: "Can you describe how you ensured your work on the chat app project was well-organized and contributed to the overall success of the project?"

**Candidate**: Sure, for the chat app project, I kept my code clean and modular, making it easier to manage. I used version control and created branches for different features, so it was easy to track changes. I also documented my code and tested it thoroughly, especially the WebSocket part. It helped in debugging and maintaining the app. Overall, it contributed to the project's success by keeping things organized and making it easier for the team to work together.
  - interviewer's read: 3/5

**Interviewer** [follow_up, c3, difficulty 3]: Can you give a specific example of a feature you worked on in the chat app project and how your efforts contributed to the overall success of the project?
  - builds on: "Sure, for the chat app project, I kept my code clean and modular, making it easier to manage."

**Candidate**: Sure, I worked on the real-time message sending feature using WebSockets. I made sure to handle messages efficiently so that the server didn't get overloaded. I kept the code simple and modular, which helped in debugging. This feature was crucial for the chat app, as it ensured messages were sent and received quickly. Overall, it contributed to the success by making the chat more responsive and user-friendly.
  - interviewer's read: 3/5

**Interviewer** [close, c3, difficulty 3]: Thank you for sharing your experience with us. Your detailed example of working on the real-time message sending feature in the chat app project and the steps you took to ensure it was efficient and user-friendly demonstrates your commitment to the project's success. Your contributions are highly valued.
  - builds on: "Can you give a specific example of a feature you worked on in the chat app project and how your efforts contributed to the overall success of the project?"
