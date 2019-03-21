CREATE TABLE students(
student_id serial PRIMARY KEY,
first_name VARCHAR(80) NOT NULL,
last_name VARCHAR(80) NOT NULL,
homeroon_number integer,
phone VARCHAR(20) UNIQUE NOT NULL,
email VARCHAR(355) UNIQUE,
graduation_year integer);

CREATE TABLE teachers(
  teacher_id serial PRIMARY KEY,
  first_name VARCHAR(80) NOT NULL,
  last_name VARCHAR(80) NOT NULL,
  homeroon_number integer,
  department VARCHAR(80),
  email VARCHAR(355) UNIQUE,
  phone VARCHAR(20) UNIQUE);

INSERT INTO students(first_name, last_name, homeroon_number, phone,graduation_year)
VALUES('MARK','Watney', 5, '7755551234', 2035)

INSERT INTO teachers(first_name, last_name, homeroon_number, department, email,phone)
VALUES('Jonas','Salk', 5, 'Biology', 'jsalk@school.org', '7755554321')
