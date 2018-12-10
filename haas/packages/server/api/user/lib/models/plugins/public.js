/* requires role plugin to be initialized first*/
module.exports.set = function(Schema) {
  Schema.plugin(publicPlugin);
};

function publicPlugin(schema) {
  // Public profile information
  schema
    .virtual('profile')
    .get(function getProfile() {
      return {
        'name': this.name,
        'role': this.role,
      };
    });

  // And id property is expected for many functions,
  // do not change that prop name
  // Non-sensitive info we can use to verify the user
  schema
  .virtual('token')
  .get(function getAuthData() {
    return {
      id: this._id, // the id propName cannot be changed, breaking
      name: this.name,
      role: this.role,
      username: this.username,
      email: this.email,
    };
  });
}
