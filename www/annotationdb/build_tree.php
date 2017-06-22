<?php 

function get_data($url) {
    $ch = curl_init();
    $timeout = 5;
    curl_setopt($ch, CURLOPT_URL, $url);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_CONNECTTIMEOUT, $timeout);
    $data = curl_exec($ch);
    curl_close($ch);
    return $data;
}

function create_db($sql) {
    $db = new PDO('sqlite::memory:');
    $db->exec($sql);
    return $db;
}

$url = 'http://storage.googleapis.com/annotationdb/annotationdb.sql?ignoreCache=1';
$sql = get_data($url);
$db = create_db($sql);

$qry = <<<EOT
    SELECT annotation, "Struct" AS type
    FROM docs
    UNION ALL
    SELECT annotation, type
    FROM annotations
EOT;

$types = [];
foreach($db->query($qry) as $row) {
    $types[$row['annotation']] = $row['type'];
}

$qry = <<<EOT
    SELECT annotation, text, study_link, study_title, study_data, free_text, selectable
    FROM docs
EOT;

$docs = [];
foreach($db->query($qry) as $row) {
    $docs[$row['annotation']] = [
        'text' => $row['text'],
        'study_link' => $row['study_link'],
        'study_title' => $row['study_title'],
        'study_data' => $row['study_data'],
        'free_text' => $row['free_text'],
        'selectable' => ($row['selectable'] == 1)
    ];
}

$qry = <<<EOT
    SELECT annotation, file_element, type, description
    FROM annotations
EOT;

$annotations = [];
foreach($db->query($qry) as $row) {
    $annotations[$row['annotation']] = [
        'file_element' => $row['file_element'],
        'type' => $row['type'],
        'description' => $row['description']
    ];
}

$qry = <<<EOT
    SELECT *
    FROM array_elements
EOT;

$array_elements = [];
foreach($db->query($qry) as $row) {
    $array_elements[$row['annotation']][] = [
        'name' => $row['element_name'],
        'type' => $row['element_type']
    ];
}

$qry = <<<EOT
    WITH RECURSIVE doctree AS (
    
        WITH bridge_table AS (
            SELECT parent, annotation
            FROM docs
            UNION ALL
            SELECT parent, annotation
            FROM annotations
        )

        SELECT parent, annotation
        FROM bridge_table
        WHERE annotation = 'va'
        
        UNION ALL
        
        SELECT b.parent, b.annotation
        FROM bridge_table AS b INNER JOIN doctree AS t
            ON b.parent = t.annotation
    )

    SELECT DISTINCT parent, annotation
    FROM doctree
EOT;

$hierarchy = [];
foreach($db->query($qry) as $row) {
    $hierarchy[$row['parent']][] = $row['annotation'];
}

$tree = [];
$tree[] = [
    'annotation' => 'va',
    ''
];

function build_tree(&$array) {
    global $docs;
    global $annotations;
    global $types;
    global $hierarchy;
    global $array_elements;
    foreach ($array as $key => $value) {
        $a = $value['annotation'];
        if ($types[$a] != "Struct") {
            $array[$key]['text'] = $annotations[$a]['file_element'];
            $array[$key]['type'] = $annotations[$a]['type'];
            $array[$key]['description'] = $annotations[$a]['description'];
            if (array_key_exists($a, $array_elements)) {
                $array[$key]['elements'] = $array_elements[$a];
            }
        }
        if (array_key_exists($a, $docs)) {
            $array[$key]['text'] = $docs[$a]['text'];
            $array[$key]['study_link'] = $docs[$a]['study_link'];
            $array[$key]['study_data'] = $docs[$a]['study_data'];
            $array[$key]['study_title'] = $docs[$a]['study_title'];
            $array[$key]['free_text'] = $docs[$a]['free_text'];
            $array[$key]['selectable'] = $docs[$a]['selectable'];
        }
        if (array_key_exists($a, $hierarchy)) {
            $new_nodes = [];
            foreach($hierarchy[$a] as $new_key => $new_value) {
                $new_nodes[] = ['annotation' => $new_value];
                unset($new_key);
                unset($new_value);
            }
            $array[$key]['nodes'] = $new_nodes;
            build_tree($array[$key]['nodes']);
        }
    }
}

build_tree($tree);

echo json_encode($tree[0]['nodes']);

?>