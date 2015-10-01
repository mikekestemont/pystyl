$(document).ready(function() {
    $("#import-btn").click(function() {
        var data = $('#import-settings-id').serializeArray();
        console.log(data);

        var control = document.getElementById("files-id");
control.addEventListener("change", function(event) {

    // When the control has changed, there are new files

    var i = 0,
        files = control.files,
        len = files.length;

    for (; i < len; i++) {
        console.log("Filename: " + files[i].name);
        console.log("Type: " + files[i].type);
        console.log("Size: " + files[i].size + " bytes");
    }

}, false);

        $.ajax({
          contentType: 'application/json;charset=UTF-8',
          url: 'processResults',
          data: JSON.stringify(data),
          type: 'POST',
          dataType: 'json',
          success: function (r) {
            console.log(r);
            $("#image-output").html(r['message']);
          }
        })
    });

    $("#preprocess-btn").click(function() {
        var data = $('#preprocessing-settings-id').serializeArray();
        console.log(data);

        $.ajax({
          contentType: 'application/json;charset=UTF-8',
          url: 'processResults',
          data: JSON.stringify(data),
          type: 'POST',
          dataType: 'json',
          success: function (r) {
            console.log(r);
            $("#image-output").html(r['message']);
          }
        })
    });

    $("#features-btn").click(function() {
        var data = $('#features-settings-id').serializeArray();
        console.log(data);

        $.ajax({
          contentType: 'application/json;charset=UTF-8',
          url: 'processResults',
          data: JSON.stringify(data),
          type: 'POST',
          dataType: 'json',
          success: function (r) {
            console.log(r);
            $("#image-output").html(r['message']);
          }
        })
    });

    $("#visualize-btn").click(function() {
        var data = $('#visualization-settings-id').serializeArray();
        console.log(data);

        $.ajax({
          contentType: 'application/json;charset=UTF-8',
          url: 'processResults',
          data: JSON.stringify(data),
          type: 'POST',
          dataType: 'json',
          success: function (r) {
            console.log(r);
            $("#image-output").html(r['message']);
          }
        })
    });
});




