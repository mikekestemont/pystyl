function process() {
    var data = $('#import-settings').serializeArray();
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
}

$(document).ready(function() {
    $("#maakresultaten").click(function() {
       process();
    });
});




