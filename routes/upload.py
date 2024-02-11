from flask import request

ALLOWED_EXTENSIONS = set(['csv'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            new_filename = f'{filename.split(".")[0]}_{str(datetime.now())}.csv'
            save_location = os.path.join('input', new_filename)
            file.save(save_location)

            output_file = process_csv(save_location)
            #return send_from_directory('output', output_file)
            return redirect(url_for('download'))

    return render_template('upload.html')