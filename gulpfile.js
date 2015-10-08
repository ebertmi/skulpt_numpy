var gulp = require('gulp');
var uglify = require('gulp-uglify');
var notify = require('gulp-notify');
var glob = require('glob');

var del = require('del');

var files = ['numpy/__init__.js', 'numpy/random/__init__.js'];
var dest = ['dist/numpy/', 'numpy/random/']

// run all watch tasks :)
gulp.task('default', ['clean', 'minify']);

gulp.task('minify', function () {
    files.forEach(function(f) {
        gulp.src(f, {base: './'})
            .pipe(uglify())
            .pipe(gulp.dest('./dist/'))
            .pipe(notify({ message: 'Finished minifying JavaScript'}));
    });
});


// Clean
// ----------------------------------------
gulp.task('clean', function () {
    return del(['./dist/*']);
});
