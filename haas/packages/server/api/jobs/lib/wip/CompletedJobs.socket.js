/**
 * Broadcast updates to client when the model changes
 */

'use strict';

var CompletedJobs = require('./completedJobs.model');

exports.register = function(socket) {
  CompletedJobs.schema.post('save', function (doc) 
  {
    onSave(socket, doc);
  });
  CompletedJobs.schema.post('remove', function (doc) 
  {
    onRemove(socket, doc);
  });
}

function onSave(socket, doc, cb) 
{
  socket.emit('thing:save', doc);
}

function onRemove(socket, doc, cb) 
{
  socket.emit('thing:remove', doc);
}