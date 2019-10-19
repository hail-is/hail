$(document).ready(function () {
    var navHeight = $('nav').height();

    if (window.location.hash) {
        var hash = window.location.hash

        var hashName = hash.substring(1, hash.length);
        var elem = document.getElementById(hashName);

        if (!elem) {
            return;
        }

        window.scrollTo(0, parseInt($(elem).offset().top, 10) - navHeight);
    }

    $('.wy-nav-content').on('click', 'a', function (e) {
        var hashName = this.href.split('#');

        if (hashName.length == 1) {
            return;
        }

        var elem = document.getElementById(hashName[1]);

        if (!elem) {
            return;
        }

        e.preventDefault();

        window.scrollTo(0, parseInt($(elem).offset().top, 10) - navHeight);
    })
});