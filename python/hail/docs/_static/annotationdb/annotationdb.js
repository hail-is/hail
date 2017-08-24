
function generate_query(annotations) {

	$('#hail-query').empty();

	if (annotations.length) {

		var split_paths = $.map(annotations.sort(), function(element, index){
			return [element.split('.')];
		});
		var display = [split_paths[0]];

		$.each(split_paths.slice(1), function(index, value){

			var last = display.slice(-1)[0];
			if (last.toString() != value.slice(0, last.length).toString()){
				display.push(value);
			} 
		});

		display = "'" + $.map(display, function(element, index){
			return element.join('.');
		}).join("',<br>        '") + "'";	

	} else {
		var display = '...';
	}

	var query =
		'<pre>' +
		'    .annotate_variants_db([' +
		'<br>' +
		"        " + display +
		'<br>' +
		'    ])' +
		'</pre>';

	$('#hail-query').html(query);
}

function build_docs(nodes, parent_id, parent_level, parent_class) {

	var level = parent_level + 1;
	var current_parent = parent_id;
	var current_parent_class = parent_class;

	$.each(nodes, function(index, node) {

		var node_name = node['annotation'].split('.').slice(-1)[0];
		var node_id = current_parent + '-' + node_name;
		var node_class = current_parent_class + ' ' + node_name;

		$('#' + current_parent).append(
			
			'<div class="panel panel-default docs level' + String(level) + '">' +
		    	'<div class="panel-heading docs" id="heading-' + node_id + '">' +
	    			'<button type="button" annotation="' + node['annotation'] + '" class="btn btn-default ' + node_class + '">' +
	    				'<i class="glyphicon glyphicon-ok"></i>' +
	    			'</button>' +
		          	'<a role="button" data-toggle="collapse" href="#' + node_id + '">' +
				       	'<span class="text-expand">' + node['text'] + '</span>' +
				   	'</a>' +
			   	'</div>' +
			   	'<div class="panel-collapse collapse" id="' + node_id + '">' +
			      	'<div class="panel-body">' +
					'</div>' +
				'</div>' +
			'</div>'
		);

		if (node['free_text']) {

			var paragraphs = node['free_text'].split("\n\n");
			var insert = '<div class="panel-text">';

			$.each(paragraphs, function(_, paragraph) {
				insert += '<p>' + paragraph.replace('\n', '<br>') + '</p>'
			});

			$('#' + node_id + '>.panel-body').append(insert + '</div>');
		}

		if (node['study_title'] && node['study_link']) {
			$('#' + node_id + '>.panel-body').append(
				'<div class="panel-text">' + 
				   '<p><a href="' + node['study_link'] + '" target="_blank">' + node['study_title'] + '</a></p>' +
				'</div>'
			);
		}
		
		if (node['study_data']) {
			$('#' + node_id + '>.panel-body').append(
				'<div class="panel-text">' +
				   '<p><a href="' + node['study_data'] + '" target="_blank">Data source</p></a>' +
				'</div>'
			);
		}

		var more_levels = node.hasOwnProperty('nodes') ? node['nodes'][0].hasOwnProperty('nodes') : false;

		if (more_levels) {

			build_docs(nodes = node['nodes'], parent_id = node_id, parent_level = level, parent_class = node_class);

		} else {

			var table_id = 'table-' + node_id;

			$('#' + node_id + '>.panel-body').append(
				'<table id="' + table_id + '" class="table table-hover" data-detail-view="true">' +
					'<thead>' + 
						'<th class="col0"></th>' +
						'<th class="col1">Variable</th>' +
						'<th class="col2">Type</th>' +
						'<th class="col3">Description</th>' +
					'</thead>' +
					'<tbody>' +
					'</tbody>' +
				'</table>'
			);

			if (!node.hasOwnProperty('nodes')) {

				$('#' + table_id + '>tbody').append(
					'<tr>' +
						'<td class="col0">' + 
							'<button type="button" annotation="' + node['annotation'] + '" class="btn btn-default ' + node_class + '">' +
			    				'<i class="glyphicon glyphicon-ok"></i>' +
			    			'</button>' +
						'</td>' +
						'<td class="col1"><span class="hail-code">' + node['annotation'] + '</span></td>' +
						'<td class="col2"><span class="hail-code">' + node['type'] + '</span></td>' +
						'<td class="col3"><span>' + node['description'] + '</span></td>' +
					'</tr>'
				);

			} else {

				$.each(node['nodes'], function(_, child) {

					var child_class = node_class + ' ' + child['annotation'].split('.').slice(-1)[0];

					$('#' + table_id + '>tbody').append(
						'<tr>' +
							'<td class="col0">' +
								'<button type="button" annotation="' + child['annotation'] + '" class="btn btn-default ' + child_class + '">' +
				    				'<i class="glyphicon glyphicon-ok"></i>' +
				    			'</button>' +
				    		'</td>' +
							'<td class="col1"><span class="hail-code">' + child['annotation'] + '</span></td>' +
							'<td class="col2"><span class="hail-code">' + child['type'] + '</span></td>' +
							'<td class="col3"><span>' + child['description'] + '</span></td>' +
						'</tr>'
					);

					if (child.hasOwnProperty('elements')) {

						var sub_table_class = 'sub-table-' + child['annotation'].replace(/\./g, '-');

						$('#' + table_id + '>tbody').append(
							'<tr class="sub-table header">' +
								'<td class="col0">' +
								'</td>' +
								'<td class="col1">' + 
									'<button type="button" class="btn btn-default accordion-toggle sub-table-icon" data-toggle="collapse" data-target="tr.sub-table[annotation=&quot;' + child['annotation'] + '&quot;]">' +
										'<i class="glyphicon glyphicon-plus"></i>' +
									'</button>' +
									'<span class="hail-code">Struct Elements:</span>' +
								'</td>' +
								'<td class="col2"></td>' +
								'<td class="col3"></td>' +
							'</tr>'
						);

						$.each(child['elements'], function(index, element){
							$('#' + table_id + '>tbody').append(
								'<tr class="sub-table hide" annotation="' + child['annotation'] + '">' +
									'<td class="col0"></td>' +
									'<td class="col1"><div class="collapse" annotation="' + child['annotation'] + '"><span class="hail-code">' + element['name'] + '</span></div></td>' +
									'<td class="col2"><div class="collapse" annotation="' + child['annotation'] + '"><span class="hail-code">' + element['type'] + '</span></div></td>' +
									'<td class="col3"><div class="collapse" annotation="' + child['annotation'] + '"><span>' + element['description'] + '</span></div></td>' +
								'</tr>'
							);
						});
					}
				
				});	

			}

		}

	});
}

$.ajax({
	url: 'https://storage.googleapis.com/annotationdb/ADMIN/tree.json',
	method: 'GET',
	dataType: 'json',
	cache: false,
	beforeSend: function(request) {
 		request.setRequestHeader('access-control-expose-headers', 'access-control-allow-origin');
 		request.setRequestHeader('access-control-allow-origin', '*');
 	},
	success: function(data) {
		build_docs(nodes = data, parent_id='panel-docs', parent_level=-1, parent_class='');
	},
	error: function() {
		console.log('Error fetching tree file.');
	}
});

function select_parents(annotation) {

	var target = $('#panel-docs button[annotation="' + annotation + '"]');
	$('#panel-docs').find(target).addClass('active');

	var split = annotation.split('.');
	var parent = split.slice(0, split.length - 1);

	if (parent.length > 1) {

		var siblings = $('#panel-docs button[annotation^="' + parent.join('.') + '"]');
		console.log(siblings);

		var n_siblings = $('#panel-docs button[annotation^="' + parent.join('.') + '"]').filter(function() {
			return ($(this).attr('annotation').split('.').length === split.length);
		}).length - 1;

		var n_siblings_selected = $('#panel-docs button[annotation^="' + parent.join('.') + '"].active').filter(function() {
			return ($(this).attr('annotation').split('.').length === split.length);
		}).length - 1;

		if (n_siblings === n_siblings_selected) {
			select_parents(parent.join('.'));
		}
	}

}

function unselect_parents(annotation) {

	var target = $('#panel-docs button[annotation="' + annotation + '"]');
	$('#panel-docs').find(target).removeClass('active');

	var split = annotation.split('.');
	var parent = split.slice(0, split.length - 1);

	if (parent.length > 1) {
		unselect_parents(parent.join('.'));
	}

}

$(document).ready(function() {
	$(document).on('click', '#panel-docs button[annotation]', function() {
		
		var annotation = $(this).attr('annotation');
		
		if ($(this).hasClass('active')) {

			var targets = $('#panel-docs button[annotation^="' + annotation + '"]');
			$('#panel-docs').find(targets).removeClass('active');
			unselect_parents(annotation);

		} else {

			var targets = $('#panel-docs button[annotation^="' + annotation + '"]');
			$('#panel-docs').find(targets).addClass('active');
			select_parents(annotation);

		}

		var selections = $('#panel-docs button.active[annotation]').map(function() {
			return $(this).attr('annotation');
		}).get();
		generate_query(selections);
	});

});

$(document).ready(function() {
	setTimeout(function() {
		$('#panel-docs div.panel:first-child div.panel-heading>button[annotation]').trigger('click');
		$('#panel-docs div.panel:first-child div.panel-heading>a').trigger('click');
	}, 500);
});

$(document).ready(function() {
	$(document).on('click', 'tr.sub-table button', function() {
		var target = $(this).attr('data-target');
		$('#panel-docs').find(target).toggleClass('hide');
		$('#panel-docs').find(target).find('td div').toggleClass('in');
		$(this).find('i').toggleClass('glyphicon-plus');
		$(this).find('i').toggleClass('glyphicon-minus');
	});
});

$(document).ready(function() {
	$(document).on('click', '#annotations-clear', function() {
		generate_query([]);
	});
});

$(document).ready(function() {
	var hail_copy_btn = document.getElementById('hail-copy-btn');
	var clipboard = new Clipboard(hail_copy_btn);
	clipboard.on('success', function(e) {});
	clipboard.on('error', function(e) {});
});
