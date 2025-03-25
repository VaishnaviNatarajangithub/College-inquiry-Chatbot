from flask import render_template, request, jsonify, redirect, url_for
import requests
from flask_server import app, db
import flask_server.university
from flask_server.university.models import Holidays, Course, Student, Teacher
from chat import chatbot_response
from flask_server.university.nlp_utils import course_matcher

with app.app_context():
    db.create_all()

@app.post("/chatbot_api/")
def normal_chat():
    msg = request.get_json().get('message')
    response, tag = chatbot_response(msg)

    # Debugging information
    print(f"Input Message: {msg}, Predicted Tag: {tag}, Response: {response}")

    if tag == 'result':
        return jsonify({'response': response, 'tag': tag, 'url': 'result/'})

    if tag == 'courses':
        course = course_matcher(msg)
        if course is not None:
            course_details = Course.query.filter_by(name=course).first()
            if course_details:
                response = f"{course_details.name} takes {course_details.duration} hours."
                link = f'http://127.0.0.1:5000/courses/syllabus/{course_details.id}/'
                return jsonify({
                    'response': response, 'tag': tag,
                    "data": {
                        "filename": f"{course_details.name} syllabus",
                        "link": link
                    }
                })
            else:
                response = "Course not found."
        else:
            courses = Course.query.all()
            response = "Available courses:\n" + "\n".join(course.name for course in courses)

    if tag == "holidays":
        holiday = Holidays.query.first()
        link = f'http://127.0.0.1:5000/holidays/download/{holiday.id}/'
        response = f"Holidays for year {holiday.year} is down below."
        return jsonify({
            'response': response, 'tag': tag,
            "data": {
                "filename": holiday.file_name,
                "link": link
            }
        })

    if tag == 'faculty':
        data = requests.get(url='http://127.0.0.1:5000/teachers/api/')
        teachers = "\n".join(f"{item['name']} ({item['department']})" for item in data.json())
        response = f"Faculty members:\n{teachers}"

    return jsonify({'response': response, 'tag': tag})

@app.post("/chatbot_api/result/")
def fetch_result():
    msg = request.get_json().get('message')
    try:
        studentID = msg.strip()
        student = Student.query.get(studentID)
        if student:
            response = f"Result of {studentID} is {student.cgpa}"
        else:
            response = "Student not found."
            url = ""
    except ValueError:
        response = "Please use the correct format : \n434121010021"
        url = "result/"
    except Exception as e:
        return jsonify({'response': "An error occurred: " + str(e), 'url': ""})

    return jsonify({'response': response, 'url': ""})
