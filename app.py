from flask import Flask, render_template, request, redirect, url_for
import os

from werkzeug.utils import secure_filename

from clustering import hierarchical_clustering


app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


# Kiểm tra định dạng file hình ảnh
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/clear_uploads')
def clear_uploads():
    folder = app.config['UPLOAD_FOLDER']
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    return redirect(url_for('upload_file'))


# Trang chủ tải ảnh
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        files = request.files.getlist('file[]')
        image_paths = []
        n_clusters = int(request.form['n_clusters'])  # Lấy số cụm từ người dùng
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                image_paths.append(file_path)

        # Thực hiện phân cụm với số lượng cụm người dùng chỉ định
        if image_paths:
            clusters, dendrogram_path = hierarchical_clustering(image_paths, n_clusters)
            return render_template('result.html', clusters=clusters, dendrogram_path=dendrogram_path)

    return render_template('index.html')



# Chạy server
if __name__ == '__main__':
    app.run(debug=True)

