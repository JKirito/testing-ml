import pandas as pd
import numpy as np
import uuid
from faker import Faker
import random
from enum import Enum
import os
from datetime import datetime, timedelta

# Initialize Faker
fake = Faker()

# Enumerations
class ProficiencyLevel(Enum):
  Beginner = 'Beginner'
  Intermediate = 'Intermediate'
  Advanced = 'Advanced'
  Expert = 'Expert'

class SkillCategoryEnum(Enum):
  Primary = 'Primary'
  Secondary = 'Secondary'

# Static Data Definitions
skill_categories = [
  {'category_id': str(uuid.uuid4()), 'category_name': 'Programming Languages', 'parent_id': None},
  {'category_id': str(uuid.uuid4()), 'category_name': 'Web Development', 'parent_id': None},
  {'category_id': str(uuid.uuid4()), 'category_name': 'Data Science', 'parent_id': None},
  {'category_id': str(uuid.uuid4()), 'category_name': 'Project Management', 'parent_id': None},
  {'category_id': str(uuid.uuid4()), 'category_name': 'DevOps', 'parent_id': None},
  # Add more categories as needed
]

skills = [
  {'skill_id': str(uuid.uuid4()), 'skill_name': 'Python Programming', 'description': 'Ability to write Python code', 'category_name': 'Programming Languages'},
  {'skill_id': str(uuid.uuid4()), 'skill_name': 'JavaScript', 'description': 'Client-side scripting language', 'category_name': 'Programming Languages'},
  {'skill_id': str(uuid.uuid4()), 'skill_name': 'Django', 'description': 'Python web framework', 'category_name': 'Web Development'},
  {'skill_id': str(uuid.uuid4()), 'skill_name': 'Machine Learning', 'description': 'Ability to create ML models', 'category_name': 'Data Science'},
  {'skill_id': str(uuid.uuid4()), 'skill_name': 'Project Management', 'description': 'Managing projects effectively', 'category_name': 'Project Management'},
  {'skill_id': str(uuid.uuid4()), 'skill_name': 'Docker', 'description': 'Containerization platform', 'category_name': 'DevOps'},
  {'skill_id': str(uuid.uuid4()), 'skill_name': 'SQL', 'description': 'Structured Query Language for database management', 'category_name': 'Programming Languages'},
  {'skill_id': str(uuid.uuid4()), 'skill_name': 'React', 'description': 'JavaScript library for building user interfaces', 'category_name': 'Web Development'},
  {'skill_id': str(uuid.uuid4()), 'skill_name': 'TensorFlow', 'description': 'Machine learning framework', 'category_name': 'Data Science'},
  {'skill_id': str(uuid.uuid4()), 'skill_name': 'Kubernetes', 'description': 'Container orchestration system', 'category_name': 'DevOps'},
  # Add more skills as needed
]

# Mapping skill_name to skill_id
skill_name_to_id = {skill['skill_name']: skill['skill_id'] for skill in skills}

# Industries and Company Sizes
industries = ['Technology', 'Healthcare', 'Finance', 'Education', 'Manufacturing', 'Retail', 'Consulting', 'Energy']
company_sizes = ['1-50', '51-200', '201-500', '501-1000', '1001+']

def generate_employers(num_employers=100):
  employers = []
  for _ in range(num_employers):
      employers.append({
          'employer_id': str(uuid.uuid4()),
          'company_name': fake.company(),
          'industry': random.choice(industries),
          'size': random.choice(company_sizes),
          'location': fake.city(),
          'website': fake.url(),
          'created_at': fake.date_between(start_date='-5y', end_date='today').isoformat(),
          'updated_at': fake.date_between(start_date='-1y', end_date='today').isoformat(),
      })
  return pd.DataFrame(employers)

def generate_candidates(num_candidates=1000):
  candidates = []
  for _ in range(num_candidates):
      candidates.append({
          'candidate_id': str(uuid.uuid4()),
          'first_name': fake.first_name(),
          'last_name': fake.last_name(),
          'email': fake.unique.email(),
          'phone': fake.phone_number(),
          'location': fake.city(),
          'preferred_job_type': random.choice(['Full-time', 'Part-time', 'Contract', 'Temporary', 'Internship']),
          'desired_salary': round(random.uniform(50000, 150000), 2),
          'created_at': fake.date_between(start_date='-5y', end_date='today').isoformat(),
          'updated_at': fake.date_between(start_date='-1y', end_date='today').isoformat(),
      })
  return pd.DataFrame(candidates)

def generate_demographics(candidates_df):
    genders = ['Male', 'Female', 'Non-Binary', 'Prefer not to say']
    ethnicities = ['Asian', 'African American', 'Caucasian', 'Hispanic', 'Other']
    demographics = []
    for _, candidate in candidates_df.iterrows():
        demographics.append({
            'demographic_id': str(uuid.uuid4()),
            'candidate_id': candidate['candidate_id'],
            'gender': random.choices(genders, weights=[25, 25, 20, 30])[0],  # Adjusted weights
            'ethnicity': random.choices(ethnicities, weights=[20, 20, 20, 20, 20])[0],  # Adjusted weights
            'age': random.randint(18, 65),
            'disability_status': random.choices([True, False, None], weights=[5, 90, 5])[0],
            'veteran_status': random.choices([True, False, None], weights=[5, 90, 5])[0],
            'created_at': fake.date_between(start_date='-5y', end_date='today').isoformat(),
            'updated_at': fake.date_between(start_date='-1y', end_date='today').isoformat(),
        })
    return pd.DataFrame(demographics)

def generate_education(candidates_df):
  degrees = ['B.Sc.', 'M.Sc.', 'Ph.D.', 'B.A.', 'M.A.']
  fields_of_study = ['Computer Science', 'Business', 'Engineering', 'Mathematics', 'Biology', 'Economics', 'Psychology', 'Marketing']
  education = []
  for _, candidate in candidates_df.iterrows():
      num_education = random.choice([1, 2])
      for _ in range(num_education):
          start_year = random.randint(1995, 2020)
          end_year = start_year + random.randint(3,6)
          education.append({
              'education_id': str(uuid.uuid4()),
              'candidate_id': candidate['candidate_id'],
              'degree': random.choice(degrees),
              'institution': fake.company(),
              'field_of_study': random.choice(fields_of_study),
              'start_date': f"{start_year}-09-01",
              'end_date': f"{end_year}-06-01",
              'grade': f"{random.uniform(2.5, 4.0):.2f}",
              'created_at': fake.date_between(start_date='-5y', end_date='today').isoformat(),
              'updated_at': fake.date_between(start_date='-1y', end_date='today').isoformat(),
          })
  return pd.DataFrame(education)

def generate_certifications(candidates_df):
  certifications_list = [
      'AWS Certified Solutions Architect',
      'Certified Scrum Master',
      'PMP',
      'Google Data Analytics',
      'Cisco Certified Network Associate',
      'Microsoft Certified Azure Developer',
      'Certified Information Systems Security Professional',
      'Oracle Certified Professional'
  ]
  issuing_organizations = [
      'Amazon', 'Scrum Alliance', 'Project Management Institute',
      'Google', 'Cisco', 'Microsoft', 'ISC2', 'Oracle'
  ]
  certifications = []
  for _, candidate in candidates_df.iterrows():
      num_certs = random.choices([0, 1, 2, 3], weights=[50, 30, 15, 5])[0]
      for _ in range(num_certs):
          issue_year = random.randint(2010, 2023)
          expiration_year = issue_year + random.randint(1,5) if random.choice([True, False]) else None
          certifications.append({
              'certification_id': str(uuid.uuid4()),
              'candidate_id': candidate['candidate_id'],
              'certification_name': random.choice(certifications_list),
              'issuing_organization': random.choice(issuing_organizations),
              'issue_date': f"{issue_year}-01-15",
              'expiration_date': f"{expiration_year}-01-15" if expiration_year else None,
              'credential_url': fake.url(),
              'created_at': fake.date_between(start_date='-5y', end_date='today').isoformat(),
              'updated_at': fake.date_between(start_date='-1y', end_date='today').isoformat(),
          })
  return pd.DataFrame(certifications)

def generate_experience(candidates_df, skills):
  experience_titles = ['Software Engineer', 'Data Analyst', 'Project Manager', 'Web Developer', 'DevOps Engineer']
  experience_employers = []  # We'll sample from the employers later
  experience = []
  for _, candidate in candidates_df.iterrows():
      num_experiences = random.choices([0,1,2,3], weights=[5, 30, 50, 15])[0]
      for _ in range(num_experiences):
          start_year = random.randint(2010, 2022)
          end_year = start_year + random.randint(1,5)
          if end_year > 2023:
              end_year = None  # Currently employed
          responsibilities = fake.paragraph(nb_sentences=5)
          skills_used = random.sample([skill['skill_name'] for skill in skills], k=random.randint(2,5))
          experience.append({
              'experience_id': str(uuid.uuid4()),
              'candidate_id': candidate['candidate_id'],
              'job_title': random.choice(experience_titles),
              'employer': fake.company(),
              'location': fake.city(),
              'start_date': f"{start_year}-06-01",
              'end_date': f"{end_year}-05-31" if end_year else None,
              'responsibilities': responsibilities,
              'skills_used': skills_used,
              'created_at': fake.date_between(start_date='-5y', end_date='today').isoformat(),
              'updated_at': fake.date_between(start_date='-1y', end_date='today').isoformat(),
          })
  return pd.DataFrame(experience)

def generate_candidate_skills(candidates_df, skills, education_df):
  candidate_skills = []
  field_skill_map = {
      'Computer Science': ['Python Programming', 'JavaScript', 'Django', 'Machine Learning', 'SQL', 'React'],
      'Business': ['Data Analysis', 'Project Management', 'Excel', 'Marketing', 'Sales'],
      'Engineering': ['CAD Design', 'Machine Learning', 'Python Programming', 'Project Management'],
      'Mathematics': ['Data Analysis', 'Machine Learning', 'Python Programming', 'SQL'],
      'Biology': ['Data Analysis', 'Research', 'Laboratory Skills'],
      'Economics': ['Data Analysis', 'Project Management', 'Excel', 'Research'],
      'Psychology': ['Research', 'Data Analysis', 'Communication Skills'],
      'Marketing': ['SEO', 'Data Analysis', 'Communication Skills', 'Project Management']
  }
  
  for _, candidate in candidates_df.iterrows():
      edu_record = education_df[education_df['candidate_id'] == candidate['candidate_id']]
      if not edu_record.empty:
          field = edu_record.iloc[0]['field_of_study']
          relevant_skills = field_skill_map.get(field, [])
          primary_skills = random.sample(relevant_skills, k=min(3, len(relevant_skills)))
      else:
          # Assign random primary skills if no education record
          primary_skills = random.sample([s['skill_name'] for s in skills], k=3)
      
      available_skills = [s['skill_name'] for s in skills if s['skill_name'] not in primary_skills]
      secondary_skills = random.sample(available_skills, k=min(2, len(available_skills)))
      
      for skill in primary_skills:
          candidate_skills.append({
              'candidate_skill_id': str(uuid.uuid4()),
              'candidate_id': candidate['candidate_id'],
              'skill_id': skill_name_to_id.get(skill, str(uuid.uuid4())),
              'proficiency_level': random.choices(
                  list(ProficiencyLevel), 
                  weights=[20, 40, 30, 10]
              )[0].value,
              'years_experience': round(random.uniform(1,10), 2),
              'skill_category': SkillCategoryEnum.Primary.value,
              'importance_weight': round(random.uniform(0.7, 1.0), 2),
              'certifications': random.choices(
                  ['AWS Certified Solutions Architect', 'Certified Scrum Master', 'PMP', 'Google Data Analytics', 'Cisco Certified Network Associate'],
                  k=random.randint(0,2)
              ),
              'last_used_date': fake.date_between(start_date='-1y', end_date='today').isoformat(),
              'created_at': fake.date_between(start_date='-5y', end_date='today').isoformat(),
              'updated_at': fake.date_between(start_date='-1y', end_date='today').isoformat(),
          })
      
      for skill in secondary_skills:
          candidate_skills.append({
              'candidate_skill_id': str(uuid.uuid4()),
              'candidate_id': candidate['candidate_id'],
              'skill_id': skill_name_to_id.get(skill, str(uuid.uuid4())),
              'proficiency_level': random.choices(
                  list(ProficiencyLevel), 
                  weights=[10, 30, 40, 20]
              )[0].value,
              'years_experience': round(random.uniform(0.5,8), 2),
              'skill_category': SkillCategoryEnum.Secondary.value,
              'importance_weight': round(random.uniform(0.3, 0.7), 2),
              'certifications': random.choices(
                  ['AWS Certified Solutions Architect', 'Certified Scrum Master', 'PMP', 'Google Data Analytics', 'Cisco Certified Network Associate'],
                  k=random.randint(0,1)
              ),
              'last_used_date': fake.date_between(start_date='-1y', end_date='today').isoformat(),
              'created_at': fake.date_between(start_date='-5y', end_date='today').isoformat(),
              'updated_at': fake.date_between(start_date='-1y', end_date='today').isoformat(),
          })
  return pd.DataFrame(candidate_skills)

def generate_jobs(employers_df, skills, num_jobs=500):
  job_titles = ['Software Engineer', 'Data Scientist', 'Product Manager', 'DevOps Engineer', 'Web Developer', 'Data Analyst', 'Project Manager', 'Systems Administrator']
  job_descriptions = [
      'Responsible for developing and maintaining software applications.',
      'Analyze data to gain insights and build predictive models.',
      'Oversee product development from conception to launch.',
      'Manage infrastructure and deployment pipelines.',
      'Design and implement web interfaces and user experiences.',
      'Collect and analyze data to support business decisions.',
      'Plan and execute projects ensuring timely delivery.',
      'Maintain and troubleshoot IT systems and networks.'
  ]
  
  jobs = []
  for _ in range(num_jobs):
      employer = employers_df.sample(1).iloc[0]
      posted_date = fake.date_between(start_date='-1y', end_date='today')
      closing_date = fake.date_between(start_date=posted_date, end_date=posted_date + timedelta(days=30))
      jobs.append({
          'job_id': str(uuid.uuid4()),
          'employer_id': employer['employer_id'],
          'title': random.choice(job_titles),
          'description': random.choice(job_descriptions),
          'location': employer['location'],
          'job_type': random.choice(['Full-time', 'Part-time', 'Contract', 'Temporary', 'Internship']),
          'salary_range': round(random.uniform(40000, 150000), 2),
          'posted_date': posted_date.isoformat(),
          'closing_date': closing_date.isoformat(),
          'created_at': posted_date.isoformat(),
          'updated_at': fake.date_between(start_date=posted_date, end_date='today').isoformat(),
      })
  return pd.DataFrame(jobs)

def generate_job_skills(jobs_df, skills):
  job_skills = []
  for _, job in jobs_df.iterrows():
      num_mandatory = random.randint(2,4)
      num_desired = random.randint(1,3)
      mandatory_skills = random.sample([s['skill_name'] for s in skills], k=num_mandatory)
      remaining_skills = [s['skill_name'] for s in skills if s['skill_name'] not in mandatory_skills]
      num_desired = min(num_desired, len(remaining_skills))
      desired_skills = random.sample(remaining_skills, k=num_desired)
      
      for skill in mandatory_skills:
          job_skills.append({
              'job_skill_id': str(uuid.uuid4()),
              'job_id': job['job_id'],
              'skill_id': skill_name_to_id.get(skill, str(uuid.uuid4())),
              'requirement_type': 'Mandatory',
              'importance_weight': round(random.uniform(0.7, 1.0), 2),
              'proficiency_level_required': random.choices(
                  list(ProficiencyLevel), 
                  weights=[10, 40, 40, 10]
              )[0].value,
              'experience_required': round(random.uniform(1,5), 2),
              'created_at': fake.date_between(start_date='-1y', end_date='today').isoformat(),
              'updated_at': fake.date_between(start_date='-1y', end_date='today').isoformat(),
          })
      
      for skill in desired_skills:
          job_skills.append({
              'job_skill_id': str(uuid.uuid4()),
              'job_id': job['job_id'],
              'skill_id': skill_name_to_id.get(skill, str(uuid.uuid4())),
              'requirement_type': 'Desired',
              'importance_weight': round(random.uniform(0.3, 0.7), 2),
              'proficiency_level_required': random.choices(
                  list(ProficiencyLevel), 
                  weights=[20, 50, 20, 10]
              )[0].value,
              'experience_required': round(random.uniform(0.5,3), 2),
              'created_at': fake.date_between(start_date='-1y', end_date='today').isoformat(),
              'updated_at': fake.date_between(start_date='-1y', end_date='today').isoformat(),
          })
  return pd.DataFrame(job_skills)

def generate_relevancy_metrics(job_skills_df):
  relevancy_metrics = []
  for _, js in job_skills_df.iterrows():
      relevancy_metrics.append({
          'metric_id': str(uuid.uuid4()),
          'job_id': js['job_id'],
          'skill_id': js['skill_id'],
          'relevancy_score': js['importance_weight'],
          'criteria_description': f"Relevancy based on {js['requirement_type']} requirement.",
          'created_at': js['created_at'],
          'updated_at': js['updated_at'],
      })
  return pd.DataFrame(relevancy_metrics)

def generate_feedback(candidates_df, employers_df, num_feedback=500):
  feedback_sources = ['Candidate', 'Employer']
  feedback = []
  for _ in range(num_feedback):
      source = random.choice(feedback_sources)
      if source == 'Candidate':
          source_id = random.choice(candidates_df['candidate_id'].tolist())
      else:
          source_id = random.choice(employers_df['employer_id'].tolist())
      
      feedback.append({
          'feedback_id': str(uuid.uuid4()),
          'source': source,
          'source_id': source_id,
          'feedback_text': fake.paragraph(nb_sentences=3),
          'rating': random.randint(1,5),
          'submitted_at': fake.date_between(start_date='-1y', end_date='today').isoformat(),
          'resolved': random.choice([True, False]),
      })
  return pd.DataFrame(feedback)

def generate_skills_data():
  # Create Skill Categories DataFrame
  skill_categories_df = pd.DataFrame(skill_categories)
  
  # Create Skills DataFrame
  skills_df = pd.DataFrame(skills)
  
  # Map category_name to category_id
  category_name_to_id = {cat['category_name']: cat['category_id'] for cat in skill_categories}
  skills_df['category_id'] = skills_df['category_name'].map(category_name_to_id)
  skills_df = skills_df.drop(columns=['category_name'])
  
  return skill_categories_df, skills_df

def main():
  # Create output directory if it doesn't exist
  output_dir = 'synthetic_data'
  os.makedirs(output_dir, exist_ok=True)
  
  # Generate Skills Data
  skill_categories_df, skills_df = generate_skills_data()
  skills_df.to_csv(os.path.join(output_dir, 'skills.csv'), index=False)
  skill_categories_df.to_csv(os.path.join(output_dir, 'skill_categories.csv'), index=False)
  print("Skills and Skill Categories generated.")
  
  # Generate Employers
  employers_df = generate_employers(num_employers=100)
  employers_df.to_csv(os.path.join(output_dir, 'employers.csv'), index=False)
  print("Employers generated.")
  
  # Generate Candidates
  candidates_df = generate_candidates(num_candidates=1000)
  candidates_df.to_csv(os.path.join(output_dir, 'candidates.csv'), index=False)
  print("Candidates generated.")
  
  # Generate Demographics
  demographics_df = generate_demographics(candidates_df)
  demographics_df.to_csv(os.path.join(output_dir, 'demographics.csv'), index=False)
  print("Demographics generated.")
  
  # Generate Education
  education_df = generate_education(candidates_df)
  education_df.to_csv(os.path.join(output_dir, 'education.csv'), index=False)
  print("Education records generated.")
  
  # Generate Certifications
  certifications_df = generate_certifications(candidates_df)
  certifications_df.to_csv(os.path.join(output_dir, 'certifications.csv'), index=False)
  print("Certifications generated.")
  
  # Generate Experience
  experience_df = generate_experience(candidates_df, skills)
  experience_df.to_csv(os.path.join(output_dir, 'experience.csv'), index=False)
  print("Experience records generated.")
  
  # Generate Candidate Skills
  candidate_skills_df = generate_candidate_skills(candidates_df, skills, education_df)
  candidate_skills_df.to_csv(os.path.join(output_dir, 'candidate_skills.csv'), index=False)
  print("Candidate Skills generated.")
  
  # Generate Jobs
  jobs_df = generate_jobs(employers_df, skills, num_jobs=500)
  jobs_df.to_csv(os.path.join(output_dir, 'jobs.csv'), index=False)
  print("Jobs generated.")
  
  # Generate Job Skills
  job_skills_df = generate_job_skills(jobs_df, skills)
  job_skills_df.to_csv(os.path.join(output_dir, 'job_skills.csv'), index=False)
  print("Job Skills generated.")
  
  # Generate Relevancy Metrics
  relevancy_metrics_df = generate_relevancy_metrics(job_skills_df)
  relevancy_metrics_df.to_csv(os.path.join(output_dir, 'relevancy_metrics.csv'), index=False)
  print("Relevancy Metrics generated.")
  
  # Generate Feedback
  feedback_df = generate_feedback(candidates_df, employers_df, num_feedback=500)
  feedback_df.to_csv(os.path.join(output_dir, 'feedback.csv'), index=False)
  print("Feedback and Ratings generated.")
  
  print("\nAll data generation complete. Files saved to the 'synthetic_data' directory.")

if __name__ == "__main__":
  main()