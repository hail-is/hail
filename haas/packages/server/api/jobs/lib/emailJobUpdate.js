const appName = process.env.APP_NAME;

if(!appName) {
  throw new Error("Require APP_NAME environemntal variable");
}

module.exports = function email(jobObj, link) {
    if (!jobObj.email) {
      return;
    }

    let subject;
    let address;
    let message;

    if(jobObj.submission.state === queueStates.completed) {
      subject = `Your job, created at ${jobObj._id.getTimestamp()}, is complete!`;
      address = `<a href="${serverAddress}/results?_id=${jobObj._id}">${appName}</a>`;
      
      message = `Visit ${address} to view or download your results`;
    } else if (jobObj.submission.state === queueStates.submitted) {
      if(jobObj.submission.attempts > 1) {
        subject = `Your job, created at ${jobObj._id.getTimestamp()} has been re-submitted`;
      } else {
        subject = `Good news! Your job, created at ${jobObj._id.getTimestamp()} has been submitted.`;
      }
      
      address = `<a href="${serverAddress}/queue?_id=${jobObj._id}">${appName}</a>`;
      
      message = `Visit ${address} to track progress`;
    } else if (jobObj.submission.state === queueStates.started) {
      if(jobObj.submission.attempts > 1) {
        subject = `Your job, created at ${jobObj._id.getTimestamp()} has been re-started`;
      } else {
        subject = `Good news! Your job, created on ${jobObj._id.getTimestamp()} has been started!`;
      }
      
      address = `<a href="${serverAddress}/queue?_id=${jobObj._id}">${appName}</a>`;
      
      message = `Visit ${address} to track progress`;
    } else if (jobObj.submission.state === queueStates.failed) {
      subject = `I'm sorry! Your job, created on ${jobObj._id.getTimestamp()} has failed`;

      address = `<a href="${serverAddress}/failed?_id=${jobObj._id}">${appName}</a>`;
      
      message = `It failed because of ${jobObj.submission.log.exceptions.join('. ')}.
       Visit ${address} to see the full job log`;
    } else if (jobObj.submission.state === queueStates.gone) {
      subject = `I'm sorry! Your job, created on ${jobObj._id.getTimestamp()} has gone missing`;

      address = `<a href="${serverAddress}/submit">${appName}</a>`;
      
      message = `We're not sure what happend. Please visit ${address} to try again`;
    }

    sqMailer.send(jobObj.email, subject, message, appName, (err) => {
      if(err) {
        log.error(`Failed to send job status update email because ${err}`);
      }
    })
  }