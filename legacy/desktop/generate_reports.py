import os
import uuid
import random
from datetime import datetime, timedelta
import string
import json
import shutil
from fpdf import FPDF
import matplotlib.pyplot as plt
import numpy as np

# Create directories if they don't exist
os.makedirs('reports', exist_ok=True)
os.makedirs('resumes', exist_ok=True)

# Sample data
first_names = ["Arjun", "Ravi", "Ananya", "Priya", "Vikram", "Neha", "Rahul", "Anjali", "Siddharth", 
               "Kavita", "Amit", "Divya", "Rajesh", "Meera", "Sunil", "Pooja", "Aditya", "Deepika", 
               "Nikhil", "Sunita", "Arun", "Shreya", "Varun", "Tanvi", "Rohit", "Nisha", "Vijay", 
               "Megha", "Anil", "Jyoti", "Sanjay", "Isha", "Rajiv", "Simran", "Kiran", "Swati"]

last_names = ["Sharma", "Patel", "Singh", "Mehta", "Gupta", "Verma", "Joshi", "Kumar", "Agarwal", 
              "Shah", "Kapoor", "Reddy", "Desai", "Malhotra", "Nair", "Menon", "Rao", "Chopra", 
              "Banerjee", "Das", "Chauhan", "Bhat", "Thakur", "Iyer", "Yadav", "Khanna", "Bose", 
              "Basu", "Chatterjee", "Mukherjee", "Roy", "Sengupta", "Chowdhury", "Gandhi", "Trivedi"]

companies = ["TechSolutions", "DataCraft", "InnovateTech", "WebWorks", "CloudMasters", 
             "CodeCrafters", "DigitalDynamics", "SmartSystems", "NexGen Solutions", 
             "FutureTech", "Infoway", "GlobalTech", "SynergySoft", "OptimusTech", 
             "VisionaryTech", "PrimeLogic", "ApexSystems", "QuantumTech", "PulseTech"]

job_titles = ["Software Engineer", "Data Scientist", "Product Manager", "UX Designer", 
              "Systems Architect", "Frontend Developer", "Backend Engineer", "DevOps Engineer", 
              "QA Engineer", "Full Stack Developer", "Mobile App Developer", "AI Specialist", 
              "Machine Learning Engineer", "Cloud Engineer", "Database Administrator"]

skills = ["Python", "Java", "C++", "JavaScript", "React", "Angular", "Node.js", "Django", 
          "Flask", "SQL", "MongoDB", "AWS", "Azure", "Docker", "Kubernetes", "Machine Learning", 
          "Data Analysis", "TensorFlow", "PyTorch", "NLP", "Computer Vision", "CI/CD", "Git", 
          "RESTful APIs", "GraphQL", "Spring Boot", "Microservices", "Unity", "Redux", "Vue.js"]

universities = ["Indian Institute of Technology", "National Institute of Technology", 
                "Delhi University", "Mumbai University", "BITS Pilani", "VIT", "Anna University", 
                "Manipal Institute of Technology", "SRM University", "Amity University", 
                "IIIT Hyderabad", "PSG College of Technology", "College of Engineering, Pune", 
                "Jadavpur University", "Thapar University"]

degrees = ["B.Tech in Computer Science", "B.E. in Information Technology", 
           "B.Sc in Computer Science", "B.Tech in Electronics", "M.Tech in Computer Science", 
           "MCA", "B.Tech in Electrical Engineering", "M.Sc in Data Science", 
           "B.E. in Computer Engineering", "MBA in IT Management", "M.Tech in AI", 
           "B.Tech in Mechanical Engineering", "B.E. in Civil Engineering"]

# Function to create a sample resume
def create_sample_resume(candidate_id, name):
    first_name, last_name = name.split()
    
    # Initialize PDF
    pdf = FPDF()
    pdf.add_page()
    
    # Set font
    pdf.set_font("Arial", size=12)
    
    # Personal Information
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, name, ln=True, align='C')
    
    pdf.set_font("Arial", size=10)
    email = f"{first_name.lower()}.{last_name.lower()}@email.com"
    phone = f"+91 {random.randint(7000000000, 9999999999)}"
    location = random.choice(["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Pune", "Kolkata"])
    
    pdf.cell(0, 5, f"Email: {email}", ln=True, align='C')
    pdf.cell(0, 5, f"Phone: {phone}", ln=True, align='C')
    pdf.cell(0, 5, f"Location: {location}, India", ln=True, align='C')
    
    # Summary
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "PROFILE SUMMARY", ln=True)
    pdf.set_font("Arial", size=10)
    
    years_exp = random.randint(1, 10)
    job_title = random.choice(job_titles)
    summary = f"Results-driven {job_title} with {years_exp} years of experience developing scalable solutions. " \
              f"Passionate about leveraging technology to solve complex business problems."
    
    pdf.multi_cell(0, 5, summary)
    
    # Skills
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "SKILLS", ln=True)
    pdf.set_font("Arial", size=10)
    
    candidate_skills = random.sample(skills, random.randint(5, 8))
    pdf.multi_cell(0, 5, ", ".join(candidate_skills))
    
    # Work Experience
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "WORK EXPERIENCE", ln=True)
    
    num_jobs = random.randint(1, 3)
    end_date = datetime.now()
    
    for i in range(num_jobs):
        company = random.choice(companies)
        job_title = random.choice(job_titles)
        
        # Calculate dates
        start_date = end_date - timedelta(days=random.randint(365, 1095))
        start_str = start_date.strftime("%b %Y")
        end_str = "Present" if i == 0 else end_date.strftime("%b %Y")
        
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(0, 8, f"{job_title} | {company}", ln=True)
        pdf.set_font("Arial", 'I', 10)
        pdf.cell(0, 5, f"{start_str} - {end_str}", ln=True)
        pdf.set_font("Arial", size=10)
        
        # Responsibilities
        resp1 = f"Developed and maintained {'web applications' if 'Web' in job_title else 'software solutions'} using {', '.join(random.sample(candidate_skills, 2))}"
        resp2 = f"{'Collaborated with cross-functional teams' if random.random() > 0.5 else 'Led a team of developers'} to deliver high-quality products"
        resp3 = f"{'Optimized' if random.random() > 0.5 else 'Improved'} {'application performance' if random.random() > 0.5 else 'system architecture'} resulting in {'faster response times' if random.random() > 0.5 else 'reduced costs'}"
        
        pdf.cell(0, 5, f"- {resp1}", ln=True)
        pdf.cell(0, 5, f"- {resp2}", ln=True)
        pdf.cell(0, 5, f"- {resp3}", ln=True)
        
        pdf.ln(3)
        end_date = start_date - timedelta(days=random.randint(30, 90))
    
    # Education
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "EDUCATION", ln=True)
    
    university = random.choice(universities)
    degree = random.choice(degrees)
    grad_year = datetime.now().year - random.randint(1, 10)
    
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 8, degree, ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 5, f"{university}, {grad_year}", ln=True)
    
    # Projects
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "PROJECTS", ln=True)
    
    project_types = ["E-commerce Platform", "Portfolio Website", "Inventory Management System", 
                     "Social Media Application", "Machine Learning Model", "Mobile Application", 
                     "Data Visualization Dashboard", "Chat Application", "Task Management Tool"]
    
    for _ in range(2):
        project = random.choice(project_types)
        tech_used = ", ".join(random.sample(candidate_skills, min(3, len(candidate_skills))))
        
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(0, 8, project, ln=True)
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 5, f"Technologies: {tech_used}", ln=True)
        pdf.multi_cell(0, 5, f"Developed a {project.lower()} that {'improves user experience' if random.random() > 0.5 else 'optimizes performance'} through {'innovative design' if random.random() > 0.5 else 'efficient implementation'}.")
        pdf.ln(3)
    
    # Save PDF
    resume_path = f"resumes/Resume_{name.replace(' ', '_')}_{candidate_id}.pdf"
    pdf.output(resume_path)
    return resume_path

# Function to generate visualization images
def create_visualizations(candidate_id):
    # Create radar chart
    categories = ['Technical Skills', 'Communication', 'Problem Solving', 
                 'Teamwork', 'Leadership', 'Domain Knowledge']
    
    values = [random.uniform(3.0, 5.0) for _ in range(len(categories))]
    
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    categories += categories[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2, linestyle='solid', color='#2271b1')
    ax.fill(angles, values, alpha=0.25, color='#2271b1')
    ax.set_thetagrids(np.degrees(angles[:-1]), categories[:-1])
    ax.set_ylim(0, 5)
    ax.set_title("Performance Assessment", size=20, pad=20)
    ax.grid(True)
    
    radar_path = f"reports/radar_{candidate_id}.png"
    plt.savefig(radar_path)
    plt.close()
    
    # Create similarity matrix
    matrix_data = np.random.uniform(0.5, 1.0, (5, 5))
    np.fill_diagonal(matrix_data, 1.0)
    
    # Make the matrix symmetric
    for i in range(5):
        for j in range(i+1, 5):
            matrix_data[j, i] = matrix_data[i, j]
            
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix_data, cmap='Blues', vmin=0, vmax=1)
    
    # Add labels
    questions = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    ax.set_xticks(np.arange(len(questions)))
    ax.set_yticks(np.arange(len(questions)))
    ax.set_xticklabels(questions)
    ax.set_yticklabels(questions)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Similarity Score", rotation=-90, va="bottom")
    
    # Add title
    ax.set_title("Response Consistency Matrix")
    
    # Save the figure
    matrix_path = f"reports/matrix_{candidate_id}.png"
    plt.savefig(matrix_path)
    plt.close()
    
    # Create gauge chart for engagement
    engagement_score = random.uniform(75, 99)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Draw the gauge
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.add_patch(plt.Circle((5, 0), 4, color='lightgray', fill=True))
    
    # Add colored arcs
    theta = np.linspace(np.pi, 0, 100)
    x = 5 + 4 * np.cos(theta)
    y = 0 + 4 * np.sin(theta)
    
    # Red zone (0-60%)
    red_end = 60 / 100 * len(theta)
    ax.plot(x[:int(red_end)], y[:int(red_end)], color='red', linewidth=15, alpha=0.7)
    
    # Yellow zone (60-80%)
    yellow_end = 80 / 100 * len(theta)
    ax.plot(x[int(red_end):int(yellow_end)], y[int(red_end):int(yellow_end)], color='yellow', linewidth=15, alpha=0.7)
    
    # Green zone (80-100%)
    ax.plot(x[int(yellow_end):], y[int(yellow_end):], color='green', linewidth=15, alpha=0.7)
    
    # Add needle
    needle_angle = np.pi * (1 - engagement_score / 100)
    needle_x = 5 + 3.8 * np.cos(needle_angle)
    needle_y = 0 + 3.8 * np.sin(needle_angle)
    ax.plot([5, needle_x], [0, needle_y], color='black', linewidth=2)
    ax.add_patch(plt.Circle((5, 0), 0.2, color='black', fill=True))
    
    # Add text
    ax.text(5, 2, f"Engagement: {engagement_score:.1f}%", ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    gauge_path = f"reports/gauge_{candidate_id}.png"
    plt.savefig(gauge_path)
    plt.close()
    
    return radar_path, matrix_path, gauge_path

# Function to create dummy interview report
def create_interview_report(candidate_id, name):
    # Create visualizations
    radar_path, matrix_path, gauge_path = create_visualizations(candidate_id)
    
    # Initialize PDF
    pdf = FPDF()
    pdf.add_page()
    
    # Set fonts
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "INTERVIEW ASSESSMENT REPORT", ln=True, align='C')
    
    # Candidate Profile
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Candidate Profile", ln=True)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(40, 10, "Name:", 0, 0)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, name, ln=True)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(40, 10, "Position:", 0, 0)
    pdf.set_font("Arial", size=12)
    job_title = random.choice(job_titles)
    pdf.cell(0, 10, job_title, ln=True)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(40, 10, "Date:", 0, 0)
    pdf.set_font("Arial", size=12)
    interview_date = (datetime.now() - timedelta(days=random.randint(1, 30))).strftime("%d %b %Y")
    pdf.cell(0, 10, interview_date, ln=True)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(40, 10, "Session ID:", 0, 0)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, candidate_id, ln=True)
    
    # Performance metrics
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Performance Metrics", ln=True)
    
    # Insert radar chart
    pdf.image(radar_path, x=20, y=None, w=170)
    
    # Engagement Analysis
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Engagement Analysis", ln=True)
    
    # Insert gauge chart
    pdf.image(gauge_path, x=20, y=None, w=170)
    
    # Response Consistency
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Response Consistency", ln=True)
    
    # Insert matrix chart
    pdf.image(matrix_path, x=20, y=None, w=170)
    
    # Technical Assessment
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Technical Assessment", ln=True)
    
    categories = ["Algorithm Knowledge", "Data Structures", "System Design", 
                 "Code Quality", "Problem Solving", "Domain Knowledge"]
    
    for category in categories:
        score = random.randint(3, 5)
        rating_text = f"{score}/5"
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(80, 10, category + ":", 0, 0)
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, rating_text, ln=True)
    
    # Recommendations
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Recommendations", ln=True)
    
    recommendation_status = random.choice(["Strongly Recommend", "Recommend", "Consider", "Do Not Recommend"])
    strengths = ["Strong technical skills", "Excellent problem-solving approach", 
                "Clear communication", "Good team player", "Innovative thinking"]
    areas_for_improvement = ["Could improve system design knowledge", 
                            "May benefit from more experience with large-scale systems", 
                            "Consider strengthening knowledge of cloud technologies"]
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(60, 10, "Final Recommendation:", 0, 0)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, recommendation_status, ln=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Strengths:", ln=True)
    pdf.set_font("Arial", size=12)
    
    for strength in random.sample(strengths, 3):
        pdf.cell(0, 8, f"- {strength}", ln=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Areas for Improvement:", ln=True)
    pdf.set_font("Arial", size=12)
    
    for area in random.sample(areas_for_improvement, 2):
        pdf.cell(0, 8, f"- {area}", ln=True)
    
    # Save PDF
    report_path = f"reports/Interview_Report_{name.replace(' ', '_')}_{candidate_id}.pdf"
    pdf.output(report_path)
    return report_path

# Generate 87 reports
for i in range(87):
    first_name = random.choice(first_names)
    last_name = random.choice(last_names)
    name = f"{first_name} {last_name}"
    
    # Generate a unique ID
    candidate_id = str(uuid.uuid4())[:8]
    
    # Create resume
    resume_path = create_sample_resume(candidate_id, name)
    
    # Create interview report
    report_path = create_interview_report(candidate_id, name)
    
    print(f"Generated report {i+1}/87: {report_path}")
    print(f"Generated resume {i+1}/87: {resume_path}")

print("\nCompleted generating 87 reports and resumes.") 