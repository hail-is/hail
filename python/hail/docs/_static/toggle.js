$(document).ready(function () {
    $("<p><span class='toggle-button'><span class='toggle-show'>▼ show</span><span class='toggle-hide'>▲ hide</span></span><p>").insertBefore($(".toggle-content"));
    $(".toggle-button").children(".toggle-hide").hide();
    $(".toggle-button").children(".toggle-show").show();
    $(".toggle").children(".toggle-content").hide();
    $(".toggle-button").click(function () {
        $(this).children(".toggle-show").toggle();
        $(this).children(".toggle-hide").toggle();
        $(this).parent().parent().children(".toggle-content").toggle(200);
    });
});
