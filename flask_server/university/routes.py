from .models import Teacher, Holidays, Student, Course
from flask_server import db, app
from flask import render_template, request, jsonify, redirect, url_for, Blueprint, send_file
from io import BytesIO

# ==============================
# Home Routes
# ==============================
@app.route("/")
def hello_world():
    return render_template('home.html')

@app.route("/home2/")
def hello_world2():
    return render_template('home2.html')


# ==============================
# Teachers Routes
# ==============================
@app.route("/teachers/", methods=['POST', 'GET'])
def teachers():
    if request.method == 'POST':
        first_name = request.form['firstname']
        last_name = request.form['lastname']
        department = request.form['department']

        new_teacher = Teacher(first_name=first_name, last_name=last_name, department=department)
        db.session.add(new_teacher)
        db.session.commit()
        return redirect(url_for('teachers'))

    teachers = Teacher.query.all()
    return render_template('teachers.html', teachers=teachers)

@app.route("/teachers/delete/<int:id>/")
def teachersdelete(id):
    teacher = Teacher.query.get(id)  # Corrected from Teacher().query.get to Teacher.query.get
    if teacher:
        db.session.delete(teacher)
        db.session.commit()
    return redirect(url_for('teachers'))

@app.route("/teachers/api/")
def teachers_api():
    teachers = Teacher.query.all()
    return jsonify([
        {
            "name": f"{teacher.first_name} {teacher.last_name}",
            "department": teacher.department,
        } for teacher in teachers
    ])

@app.route("/teachers/api/<string:dept>/")
def dept_teachers_api(dept):
    teachers = Teacher.query.filter(Teacher.department.ilike(f"%{dept}%")).all()
    return jsonify([
        {
            "name": f"{teacher.first_name} {teacher.last_name}",
            "department": teacher.department,
        } for teacher in teachers
    ])


# ==============================
# Holidays Routes
# ==============================
@app.route("/holidays/", methods=['POST', 'GET'])
def holidays():
    if request.method == 'POST':
        year = request.form['year']
        data = request.files['file']
        new_holiday = Holidays(year=year, file_name=data.filename, data=data.read())
        db.session.add(new_holiday)
        db.session.commit()
        print(f"{new_holiday} added")
        return redirect(url_for('holidays'))

    holidays = Holidays.query.all()
    return render_template('holidays.html', holidays=holidays)

@app.route("/holidays/download/<int:id>/")
def holidays_file_api(id):
    holiday = Holidays.query.get(id)
    if holiday:
        return send_file(BytesIO(holiday.data), download_name=holiday.file_name)
    return "Holiday file not found", 404


# ==============================
# Students Routes
# ==============================
@app.route("/students/", methods=['POST', 'GET'])
def students():
    if request.method == 'POST':
        studentID = request.form['studentID']
        name = request.form['name']
        courseID = request.form['course']

        new_student = Student(id=studentID, name=name, course_id=courseID)
        db.session.add(new_student)
        db.session.commit()
        return redirect(url_for('students'))

    students = Student.query.all()
    courses = Course.query.all()
    return render_template('students.html', students=students, courses=courses)

@app.route("/students/update/<int:id>/", methods=['POST', 'GET'])
def studentsupdate(id):
    student = Student.query.get(id)  # Corrected from Student().query.get to Student.query.get
    if not student:
        return "Student not found", 404

    if request.method == 'POST':
        student.cgpa = request.form['cgpa']
        student.name = request.form['name']
        student.id = request.form['studentID']
        db.session.commit()
        return redirect(url_for('students'))

    return render_template('student_update.html', student=student)


# ==============================
# Courses Routes
# ==============================
@app.route("/courses/", methods=['POST', 'GET'])
def courses():
    if request.method == 'POST':
        name = request.form['name']
        duration = request.form['duration']
        syllabus = request.files['file']

        new_course = Course(name=name, duration=duration, syllabus=syllabus.read())
        db.session.add(new_course)
        db.session.commit()
        return redirect(url_for('courses'))

    courses = Course.query.all()
    return render_template('courses.html', courses=courses)

@app.route("/courses/delete/<int:id>/")
def coursesdelete(id):
    course = Course.query.get(id)
    if course:
        db.session.delete(course)
        db.session.commit()
    return redirect(url_for('courses'))

@app.route("/courses/update/<int:id>/", methods=['POST', 'GET'])
def courses_update(id):
    course = Course.query.get(id)
    if not course:
        return "Course not found", 404

    if request.method == 'POST':
        syllabus_file = request.files.get('file')  # Use .get() for safe access
        if syllabus_file:
            course.syllabus = syllabus_file.read()
        # Optionally handle other fields if needed
        db.session.commit()
        return redirect(url_for('courses'))

    return render_template('course_update.html', course=course)

@app.route("/courses/syllabus/<int:id>/")
def syllabus_api(id):
    course = Course.query.get(id)
    if course:
        return send_file(BytesIO(course.syllabus), download_name=f"{course.name}.pdf")
    return "Syllabus not found", 404
