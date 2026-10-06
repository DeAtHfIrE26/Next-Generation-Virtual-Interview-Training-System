# Mock interview 03: Product Manager (mid) at Atlassian

- Type / round: behavioral / final; duration 15 min; language en; requested difficulty auto
- Company style: Values-based behavioural questions, product sense
- Skills to probe: -
- Candidate profile (simulated): average
- Interviewer LLM: ollama:qwen2.5:7b-instruct
- Checks: PASS {"questions": 9, "bank_overlap": 0, "follow_ups": 4, "follow_ups_referencing_answer": 4, "difficulty_tracks_performance": true, "difficulty_corrections_by_code": 1, "emergency_questions": 0, "fresh_questions": 0, "rejected_attempts": {"repeats an earlier question": 2}, "emergency_blueprint": false, "repeats": 0, "closed_on_time": true, "passed": true}

## Blueprint
This interview assesses the candidate's ability to own and improve a critical product feature and their customer engagement and cross-functional collaboration skills, which are essential for a mid-level Product Manager at Atlassian.

- **Ownership** (4 min): Ownership is crucial for a mid-level Product Manager to drive improvements and success in key product areas.
- **Customer Engagement** (4 min): Atlassian values understanding and addressing customer needs, and a Product Manager must actively engage with customers to ensure product success.
- **Cross-Functional Collaboration** (3 min): Collaboration with design and engineering teams is essential to align product vision and deliver successful outcomes.
- **Product Sense** (2 min): Atlassian looks for individuals who can make strategic decisions based on product intuition and data-driven insights.

## Transcript

**Interviewer** [open, c1, difficulty 3]: Hi, thank you for joining us today. Let's start with a few questions to understand your experience with owning a product feature and your approach to customer engagement. Can you tell me about a time when you were responsible for owning a product feature, what was the feature, and what was your initial goal?

**Candidate**: Sure, I can share that experience. At my previous role, I was part of the team that owned the project management features in our collaboration tool. My initial goal was to enhance the task tracking functionality to make it more intuitive for our users. We wanted to improve the visibility and manageability of tasks, especially for teams that were ramping up on the platform. By the end of the project, we had streamlined the workflow, which led to a 25% increase in user satisfaction.
  - interviewer's read: 4/5

**Interviewer** [follow_up, c1, difficulty 3]: That's great to hear. Can you elaborate on the specific actions you took to streamline the workflow and how you ensured that the 25% increase in user satisfaction was achieved?
  - builds on: "By the end of the project, we had streamlined the workflow, which led to a 25% increase in user satisfaction."

**Candidate**: Sure, to streamline the workflow, we first conducted a user feedback session to understand pain points. We then reorganized the task layout to group similar tasks together and added clear labels. We also introduced a drag-and-drop feature to make moving tasks easier. To ensure the satisfaction increase, we ran A/B tests and gathered post-implementation feedback. We also set up a metrics dashboard to monitor key performance indicators like task completion time and user engagement. The combination of these actions definitely helped us meet our goal.
  - interviewer's read: 4/5

**Interviewer** [follow_up, c1, difficulty 4]: That's excellent. Can you provide more details on the specific metrics you used to measure the success of the improvements? Were there any trade-offs or challenges you faced during this process?
  - builds on: "We also set up a metrics dashboard to monitor key performance indicators like task completion time and user engagement."

**Candidate**: Absolutely. We focused on a few key metrics: task completion time, user engagement, and the number of support tickets related to task management. We saw a 20% reduction in task completion time and a 30% increase in user engagement. Regarding challenges, we faced some resistance from users who were accustomed to the old layout, but we managed this by providing comprehensive training sessions and a clear change management plan. There was also a trade-off in initial implementation time, as we had to pause some other initiatives to focus on this project, but the long-term benefits outweighed the short-term costs.
  - interviewer's read: 4/5

**Interviewer** [new_topic, c2, difficulty 4]: That's great. Can you tell me about a time when you took a proactive approach to discover customer needs and how you engaged with them to improve a product feature?

**Candidate**: Sure, at my previous role, I noticed that users were frequently asking for better integration options with third-party tools. To address this, I organized a series of user interviews and focus groups. We also set up an internal survey to gather more quantitative data on the issue. Based on the feedback, we decided to prioritize a feature that would allow seamless integration with popular tools. We engaged with a cross-functional team, including developers and designers, to create a prototype. We then tested this prototype with a small group of users and collected their feedback. This iterative process helped us refine the feature, ensuring it met the needs of our users. The result was a 40% increase in user retention in the following quarter, which was a significant improvement.
  - interviewer's read: 4/5

**Interviewer** [follow_up, c2, difficulty 4]: Great. Can you walk us through the process of engaging with the cross-functional team and how their contributions helped in the successful implementation of the feature?
  - builds on: "Based on the feedback, we decided to prioritize a feature that would allow seamless integration with popular tools. We engaged with a cross-functional team, including developers and designers, to create a prototype."

**Candidate**: Definitely. When we identified the need for better integration options, we organized a kickoff meeting with key stakeholders from the engineering, design, and product teams. We shared the insights from our user interviews and surveys and discussed potential approaches. The engineering team was responsible for developing the integration endpoints, while the design team worked on the user interface to make it seamless. We held regular check-ins and shared progress updates to keep everyone aligned. The cross-functional collaboration was crucial; the engineering team's technical expertise helped us overcome some implementation hurdles, and the design team ensured the feature was user-friendly. This teamwork ensured that the feature was not only technically sound but also met the user's needs, leading to the successful implementation and positive user feedback.
  - interviewer's read: 4/5

**Interviewer** [follow_up, c3, difficulty 4]: That's really insightful. Can you provide specific examples of how the contributions from each team member were instrumental in the success of the feature?
  - builds on: "The cross-functional collaboration was crucial; the engineering team's technical expertise helped us overcome some implementation hurdles, and the design team ensured the feature was user-friendly."

**Candidate**: Certainly. During the development of the integration feature, the engineering team's technical expertise was crucial. They helped us set up the necessary endpoints and integrate them with our existing platform, ensuring smooth data flow between our system and third-party tools. On the design side, they created intuitive interfaces that made the integration process easy for users. For instance, they designed a wizard that guided users through the setup process step-by-step, reducing the learning curve. Additionally, our QA team conducted rigorous testing to ensure that the feature worked flawlessly across different scenarios. Their feedback helped us catch and resolve several bugs before the feature was rolled out to the wider user base. Overall, the cross-functional collaboration ensured that the feature was robust, user-friendly, and aligned with our users' needs, leading to a successful implementation.
  - interviewer's read: 4/5

**Interviewer** [new_topic, c4, difficulty 4]: Can you describe a situation where you used data and your intuition to make a strategic decision that improved a product's performance?

**Candidate**: Sure, at my previous role, we noticed that our collaboration tool was being used less frequently by teams with more than 10 members. Initially, we thought it might be due to the size of the teams, but data showed that even smaller teams were finding the tool useful. To investigate further, we conducted a survey and analyzed user sessions. We found that the issue was related to the complexity of managing tasks within the tool. The UI was too cluttered and didn't provide enough context for large teams. To address this, we decided to implement a feature that would allow users to create sub-teams within the main project. This would help break down tasks and make the tool more manageable. We tested this feature with a small group of users and the feedback was overwhelmingly positive. We then rolled it out to a larger group and saw a 30% increase in active users within large teams. This decision was based on a combination of data and user intuition, and it significantly improved the tool's performance.
  - interviewer's read: 4/5

**Interviewer** [wrap_up, c4, difficulty 4]: Thank you for sharing that detailed example. It clearly shows your ability to use both data and intuition to make strategic decisions that positively impact the product. Can you also tell me how this decision influenced the broader product ecosystem and any long-term benefits it brought?
  - builds on: "we saw a 30% increase in active users within large teams. This decision was based on a combination of data and user intuition, and it significantly improved the tool's performance."

**Candidate**: Absolutely. The decision to introduce sub-teams not only improved the task management experience for large teams but also enhanced the overall collaboration within the ecosystem. By making the tool more scalable, we allowed larger teams to effectively use our platform without overwhelming them. This, in turn, attracted more large teams to our product, which helped us grow our user base. Long-term, this feature reduced the churn rate among larger teams, as they found the tool more user-friendly and better suited to their needs. It also paved the way for future integrations and features that could further enhance the collaboration experience for all team sizes. Overall, the decision to implement sub-teams had a positive ripple effect across the product ecosystem, making it more versatile and user-friendly.
  - interviewer's read: 4/5

**Interviewer** [close, c4, difficulty 4]: Thank you for your detailed responses. It was great to hear about your experience and how you've driven success in various aspects of product management. It was a pleasure speaking with you, and we appreciate your time today. Goodbye.
  - builds on: "Can you describe a situation where you used data and your intuition to make a strategic decision that improved a product's performance?"
