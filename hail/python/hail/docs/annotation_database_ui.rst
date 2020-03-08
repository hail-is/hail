.. _Annotation Database:


===================
Annotation Database
===================

This database contains a curated collection of variant annotations in an accessible and Hail-friendly format, for use in Hail analysis pipelines.

To incorporate these annotations in your own Hail analysis pipeline, select which annotations you would like to query from the table below and then copy-and-paste the Hail generated code into your own analysis script.

--------------

Database Query
--------------

Select annotations by clicking on the checkboxes in the table, and the appropriate Hail command will be generated
in the panel below.

In addition, a search bar is provided if looking for a specific annotation within our curated collection.

Use the "Copy to clipboard" button to copy the generated Hail code, and paste the command into your
own Hail script.

.. raw:: html
    <div class="jumbotron" sytle="margin-bottom:0px; color:white;">
    <h2 class="text-center" style="font-size:40px; font-weight:400;"> Hail </h2>
    </div>

    <div class="d-flex flex-column" style="height:1150px;" id='annotation-db'>
    <div class="flex-1" >
    <div class="panel panel-default flex-fill">
    <div class="panel-heading" style="font-weight:bold">Search</div>
    <div class="panel-body">
    <div class="search">
    <div class="col-xs-6">
    <span>Type in annotation:</span>
    <input type="text" id='searchInput' onkeyup="filterTable()" name="keyword" class="form-control input-sm" placeholder="Enter Annotation....">
    </div>
    </div>
    </div>
    </div>
    <div class="panel panel-default" >
    <div class="panel-heading" style="font-weight:bold">Database Query
    <div class="btn-group pull-right" id="copy-button-container">
    <button class="btn btn-default btn-sm" onclick="copy()">Copy to Clipboard</button>
    </div>
    </div>
    <div class="panel-body">
    <div class="search ">
    <div class="col-xs-6">
    <div class="form-group">
    <label for="exampleFormControlTextarea1">Hail Generated Code</label>
    <textarea readonly class="form-control" id="result" rows="3">db = hl.experimental.DB()
    mt = db.annotate_rows_db(mt)</textarea>
    </div>
    </div>
    </div>
    </div>
    </div>
    </div>
    <div class="row" style="margin-top:10px"></div>
    <table id='table1' class="table1 table table-bordered display select" >
    <div id="header"  class="text-center"></div>
    <tr id="tableHeader">
    <th><input id="checkAll" name="addall" type="checkbox" ></th>
    <th>name</th>
    <th>description</th>
    <th>version</th>
    </tr>
    </table>
    </div>
