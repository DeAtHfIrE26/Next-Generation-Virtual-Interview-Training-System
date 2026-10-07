from fpdf import FPDF

# Create instance of FPDF class
pdf = FPDF()

# Add a page
pdf.add_page()

# Set fonts
pdf.set_font("Arial", 'B', 16)
pdf.cell(0, 10, "John Smith", 0, 1, 'C')

pdf.set_font("Arial", 'B', 12)
pdf.cell(0, 10, "Software Engineer", 0, 1, 'C')

pdf.set_font("Arial", '', 11)
pdf.cell(0, 8, "Email: john.smith@example.com | Phone: (555) 123-4567", 0, 1, 'C')
pdf.cell(0, 8, "Location: New York, NY", 0, 1, 'C')

pdf.ln(5)
pdf.set_font("Arial", 'B', 12)
pdf.cell(0, 10, "SUMMARY", 0, 1, 'L')
pdf.set_font("Arial", '', 10)
pdf.multi_cell(0, 5, "Experienced software engineer with 5+ years of expertise in developing web applications using Python, JavaScript, and various frameworks. Skilled in machine learning and data analysis. Strong background in agile methodologies and team collaboration.")

pdf.ln(5)
pdf.set_font("Arial", 'B', 12)
pdf.cell(0, 10, "SKILLS", 0, 1, 'L')
pdf.set_font("Arial", '', 10)
pdf.multi_cell(0, 5, "Programming Languages: Python, JavaScript, TypeScript, Java\nFrameworks: React, Django, Flask, TensorFlow\nTools: Git, Docker, AWS, Jenkins\nSoft Skills: Team Leadership, Communication, Problem-solving")

pdf.ln(5)
pdf.set_font("Arial", 'B', 12)
pdf.cell(0, 10, "WORK EXPERIENCE", 0, 1, 'L')

# Job 1
pdf.set_font("Arial", 'B', 10)
pdf.cell(0, 6, "Senior Software Engineer - Tech Innovations Inc.", 0, 1, 'L')
pdf.set_font("Arial", 'I', 10)
pdf.cell(0, 6, "January 2020 - Present", 0, 1, 'L')
pdf.set_font("Arial", '', 10)
pdf.multi_cell(0, 5, "- Developed and maintained cloud-based applications using Python and AWS\n- Led a team of 5 developers in implementing new features and resolving bugs\n- Reduced application load time by 40% through code optimization")

# Job 2
pdf.ln(2)
pdf.set_font("Arial", 'B', 10)
pdf.cell(0, 6, "Software Developer - DataSoft Solutions", 0, 1, 'L')
pdf.set_font("Arial", 'I', 10)
pdf.cell(0, 6, "June 2017 - December 2019", 0, 1, 'L')
pdf.set_font("Arial", '', 10)
pdf.multi_cell(0, 5, "- Built RESTful APIs using Django and Flask\n- Implemented machine learning algorithms for data classification\n- Collaborated with product managers to define and implement new features")

pdf.ln(5)
pdf.set_font("Arial", 'B', 12)
pdf.cell(0, 10, "EDUCATION", 0, 1, 'L')
pdf.set_font("Arial", 'B', 10)
pdf.cell(0, 6, "Master of Science in Computer Science", 0, 1, 'L')
pdf.set_font("Arial", '', 10)
pdf.cell(0, 6, "New York University - 2017", 0, 1, 'L')

pdf.set_font("Arial", 'B', 10)
pdf.cell(0, 6, "Bachelor of Science in Computer Engineering", 0, 1, 'L')
pdf.set_font("Arial", '', 10)
pdf.cell(0, 6, "University of California, Berkeley - 2015", 0, 1, 'L')

pdf.ln(5)
pdf.set_font("Arial", 'B', 12)
pdf.cell(0, 10, "PROJECTS", 0, 1, 'L')
pdf.set_font("Arial", 'B', 10)
pdf.cell(0, 6, "Predictive Analytics Tool", 0, 1, 'L')
pdf.set_font("Arial", '', 10)
pdf.multi_cell(0, 5, "Developed a machine learning-based tool that predicts customer behavior with 85% accuracy. Used Python, TensorFlow, and scikit-learn.")

pdf.ln(2)
pdf.set_font("Arial", 'B', 10)
pdf.cell(0, 6, "E-commerce Platform", 0, 1, 'L')
pdf.set_font("Arial", '', 10)
pdf.multi_cell(0, 5, "Created a full-stack e-commerce platform using React, Node.js, and MongoDB. Implemented secure payment processing and inventory management.")

# Save the pdf with name sample_resume.pdf
pdf.output("sample_resume.pdf")

print("Sample resume created as sample_resume.pdf") 