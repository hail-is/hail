
function generate_query() {

	// get all selected nodes from tree
	var selected_nodes = $('#tree').treeview('getSelected');

	// for each selected node, get Hail path of annotation
	var selected_hail = $.map(selected_nodes, function(value, index) {
		return value['hail'];
	});

	// if at least 1 annotation is selected, generate query to display
	if (selected_hail.length) {

		// split each annotation Hail path into its components
		var split_paths = $.map(selected_hail.sort(), function(element, index){
			return [element.split('.')];
		});

		// array of annotations to display, initialize with first annotation
		var display = [split_paths[0]];

		// picking only minimum annotations necessary to display
		$.each(split_paths.slice(1), function(index, value){

			// last annotation currently displayed
			var last = display.slice(-1)[0];

			// if current path has same beginning as last currently displayed, skip
			if (last.toString() == value.slice(0, last.length).toString()){
				return true
			} 

			// add annotation to display array
			display.push(value);

		});

		// rejoin display annotation paths
		display = $.map(display, function(element, index){
			return element.join('.');
		});

		var query = '<pre class="import-function">' +
				    '    .annotate_variants_db([' + 
				    '<br>' +
					"        '" + display.join("',<br>        '") +
					"'<br>" +
					'    ])' +
					'</pre>'

		// empty DOM element and insert newly created query
		$( 'span.hail-code.query ').empty();
		$( 'span.hail-code.query ').html(query);
	
	// if no annotations selected, clear query panel			
	} else {

		// boilerplate template
		var template = '<pre class="import-function">' +
					     '    .annotate_variants_db([' +
					     '<br>' +
					     '        ...' +
				    	 '<br>' +
				     	 '    ])' +
				       '</pre>'

		// empty DOM and insert empty string
		$( 'span.hail-code.query ').empty();
		$( 'span.hail-code.query' ).html(template);
	}
}

function select_nodes(node, selections){

	var current_selections = selections;

	if (node.hasOwnProperty('nodes')){

		children = node['nodes'];

		$.each(children, function(index, child){
			current_selections.push(child);
			current_selections = select_nodes(child, current_selections);
		});

	}

	return current_selections
}

function build_tree(data){

		// use data dictionary to build query tree using bootstrap-tree.js library
		$('#tree').treeview({
			data: data,
			multiSelect: true,
			levels: 1,
			expandIcon: 'glyphicon-plus',
			collapseIcon: 'glyphicon-minus',
			onNodeSelected: function(event, node){

				var selections = $('#tree').treeview('getSelected');
				var new_selections = select_nodes(node, selections);

				$(this).treeview('unselectNode', [ new_selections, {silent: true} ]);
				$(this).treeview('selectNode', [ new_selections, {silent: true} ]);

				generate_query();
			},
			onNodeUnselected: function(event, node){

				var unselections = select_nodes(node, []);
				var tree = $(this).treeview(true);

				$(this).treeview('selectNode', [ unselections, {silent: true} ]);
				$(this).treeview('unselectNode', [ unselections, {silent: true }]);

				generate_query();
			}
		});
}

function build_table(node, table_id){

	if (node.hasOwnProperty('nodes')){
		$.each(node['nodes'], function(_, child_node){
			build_table(child_node, table_id);
		});
	} else {

		var col0 = '';
		var col1 = node['hail'];
		var col2 = node['type'];
		var col3 = node['description'];

		var has_elements = (node.hasOwnProperty('elements') ? true : false);

		if (has_elements){

			var sub_table_id = 'sub-' + table_id + '-' + node['hail'].split('.').slice(-1)[0];

			col0 = '<a data-toggle="collapse" href="#row-' + sub_table_id + '" aria-expanded="false" aria-controls="row-' + sub_table_id + '">' +
				       '<i class="sub-table-icon glyphicon glyphicon-plus icon-plus"><i>' +
				   '</a>';

		}

		$('#' + table_id + '>tbody').append(
			'<tr>' +
				'<td class="col0">' + col0 + '</td>' +
				'<td class="col1"><span class="hail-code">' + col1 + '</span></td>' +
				'<td class="col2"><span class="hail-code">' + col2 + '</span></td>' +
				'<td class="col3"><span>' + col3 + '</span></td>' +
			'</tr>'
		);

		if (has_elements){

			$('#' + table_id + '>tbody').append(
				'<tr id="row-' + sub_table_id + '" class="collapse">' +
					'<td colspan="4" class="sub-table">' +
						'<table id="' + sub_table_id + '" class="table table-hover sub-table">' + 
							'<thead>' +
								'<th>Struct Element</th>' +
								'<th>Type</th>' +
							'</thead>' +
							'<tbody>' +
							'</tbody>' +
						'</table>' +								
					'</td>' +
				'</tr>'
			);

			$.each(node['elements'], function(_, element){
				$('#' + sub_table_id + '>tbody').append(
					'<tr>' +
						'<td class="sub-col0"><span class="hail-code">' + element['id'] + '</span></td>' +
						'<td class="sub-col1"><span class="hail-code">' + element['type'] + '</span></td>' +
					'</tr>'
				);
			});
		}
	}
}

function build_group(node, doc_id, level) {

	var node_id = node['hail'].split('.').slice(-1)[0];
	var new_doc_id = doc_id + '-' + node_id;
	var new_level = level + 1;

	$('#' + doc_id).append(
		'<div class="panel panel-default level' + String(new_level) + '">' +
	    	'<div class="panel-heading" id="heading-' + new_doc_id + '">' +
	          	'<a role="button" data-toggle="collapse" href="#' + new_doc_id + '">' +
			       	'<span class="text-expand">' + node['text'] + '</span>' +
			   	'</a>' +
		   	'</div>' +
		   	'<div class="panel-collapse collapse" id="' + new_doc_id + '">' +
		      	'<div class="panel-body">' +
				'</div>' +
			'</div>' +
		'</div>'
	);

	if (node['free_text']) {

		var paragraphs = node['free_text'].split("\n\n");
		var insert = '<div class="panel-text">'

		$.each(paragraphs, function(_, paragraph){
			insert += '<p>' + paragraph.replace('\n', '<br>') + '</p>'
		});

		$('#' + new_doc_id + '>.panel-body').append(insert + '</div>');

	}

	if (node['study_title'] && node['study_link']) {

		$('#' + new_doc_id + '>.panel-body').append(
			'<div class="panel-text">' + 
			   '<p><a href="' + node['study_link'] + '" target="_blank">' + node['study_title'] + '</a></p>' +
			'</div>'
		);

	}
	
	if (node['study_data']) {

		$('#' + new_doc_id + '>.panel-body').append(
			'<div class="panel-text">' +
			   '<p><a href="' + node['study_data'] + '" target="_blank">Data source</p></a>' +
			'</div>'
		);

	}

	if (!node.hasOwnProperty('db_file')){

		$.each(node['nodes'], function(_, child_node){
			build_group(node=child_node, doc_id=new_doc_id, level=new_level);
		});

	} else {

		var table_id = 'table-' + new_doc_id;

		$('#' + new_doc_id + '>.panel-body').append(
			'<table id="' + table_id + '" class="table table-hover" data-detail-view="true">' +
				'<thead>' + 
					'<th class="col0 detail"></th>' +
					'<th class="col1">Variable</th>' +
					'<th class="col2">Type</th>' +
					'<th class="col3">Description</th>' +
				'</thead>' +
				'<tbody>' +
				'</tbody>' +
			'</table>'
		);

		build_table(node=node, table_id=table_id);
		
	} 
}

function build_docs(data) {
	$.each(data, function(index, node) {
		build_group(node=node, doc_id='panel-docs', level=-1);
	});
}
	
$.ajax({
	url: 'https://storage.googleapis.com/annotationdb/tree.json',
	method: 'GET',
	dataType: 'json',
	cache: false,
	beforeSend: function(request) {
		request.setRequestHeader('access-control-expose-headers', 'access-control-allow-origin');
		request.setRequestHeader('access-control-allow-origin', '*');
	},
	success: function(data) {

		/* data =
			[
				{
					"text": "CADD",
					"hail": "va.cadd",
					...
					"nodes": [
						{
							"text": "Raw score",
							"hail": "va.cadd.RawScore",
							"type": "Double",
							"description": "Raw CADD score"
						},
						...
					]
				},
				{
					"text": "DANN",
					"hail": "va.dann",
					...,
					"nodes": [
						...
					]
				},
				...
	    	]
 		*/

		// use data dictionary to build query tree
		build_tree(data);

		// use data dictionary to build documentation
		build_docs(data);
	},
	error: function() {
		window.alert('Error fetching tree file!');
	}
});

$(document).ready(function(){
	$(document).on('click', 'i.sub-table-icon', function(e) {
		$(this).toggleClass('glyphicon-plus icon-plus');
		$(this).toggleClass('glyphicon-minus icon-minus');
	});
});

// functionality to clear selection when button is clicked
$('#annotations-clear').on('click', function() {
	var tree = $('#tree').treeview(true);
	var selected = tree.getSelected();
	if (selected.length){
		tree.unselectNode(selected, {silent: true});
		generate_query();
	}
});

// copy query to clipboard when button is clicked
var hail_copy_btn = document.getElementById('hail-copy-btn');
var clipboard = new Clipboard(hail_copy_btn);
clipboard.on( 'success', function(e) {});
clipboard.on( 'error', function(e) {});
