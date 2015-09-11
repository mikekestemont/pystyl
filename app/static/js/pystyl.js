function process() {
    var data = $('#settings').serializeArray();
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
    $('#pystyl-settings').hide();
    $('#visualize-settings').hide();

    $("#maakresultaten").click(function() {
       process();
    });

    $("#import").click(function() {
       $('#pystyl-settings').hide();
       $('#visualize-settings').hide();
       $('#import-settings').show();
       console.log("Import clicked");
    });

    $("#pystylize").click(function() {
       $('#import-settings').hide();
       $('#visualize-settings').hide();
       $('#pystyl-settings').show();
       console.log("Pystylize clicked");
    });

    $("#visualize").click(function() {
       $('#import-settings').hide();
       $('#pystyl-settings').hide();
       $('#visualize-settings').show();
       console.log("Visualize clicked");
    });

});
