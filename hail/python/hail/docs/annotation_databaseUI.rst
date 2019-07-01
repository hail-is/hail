.. raw:: html   
    <div class="jumbotron" sytle="margin-bottom:0px; color:white;">
    <h2 class="text-center" style="font-size:40px; font-weight:400;"> Hail </h2>
    </div>  
      
    <div class="container" style="height:1150px;">
    <div class="row" >
    <div class="panel panel-default">
    <div class="panel-heading" style="font-weight:bold">Search</div>
    <div class="panel-body">
    <div class="search row">
    <div class="col-xs-6">
    <span>Type in annotation:</span>
    <input type="text" id='searchInput' onkeyup="searchTable()" name="keyword" class="form-control input-sm" placeholder="Enter Annotation....">
    </div>
    </div>
    </div>
    </div>
    <div class="panel panel-default">
    <div class="panel-heading" style="font-weight:bold">Database Query
    <div class="btn-group pull-right">
    <a href="#" class="btn btn-default btn-sm" onclick="copy()" style="height: 27px; font-weight:22">Copy to Clipboard</a>
    </div>
    </div>
    <div class="panel-body">
    <div class="search row">
    <div class="col-xs-6">
    <div class="form-group">
    <label for="exampleFormControlTextarea1">Hail Generated Code</label>
    <textarea readonly class="form-control" id="result" rows="3" width="500px" style="font-family:monospace">db = hl.experimental.DB()      
    mt = db.annotate_rows_db(mt, </textarea>
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
    <th>version</th>
    <th>n_rows</th>
    <th>n_cols</th>
    </tr>
    </table>
    </div>