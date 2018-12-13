const uuid = require('uuid');
const log = require.main.require('./common/logger');

module.exports.set = function set(Schema, options) {
  Schema.plugin(rolesPlugin, options);
};

function rolesPlugin(schema, options) {
  /* Roles*/
  const _roles = options.roles;

  /* expose to other plugins and consumers*/
  schema.statics.roles = _roles;

  schema.add({
    role: {
      type: String,
      default: _roles[1], // default and enum are reserve words
      enum: _roles
    }
  });
  /**
   * Static methods
   */
  schema.statics.makeGuest = function makeGuest() {
    return {
      id: _roles[0] + uuid.v4(), // pseudo-random, rare collisions
      name: _roles[0],
      role: _roles[0] // not a real user
    };
  };

  schema.statics.roleExists = function roleExists(user) {
    return !!(user && user.role);
  };
  /* Guest role is always lowest index */
  schema.statics.isGuest = function isGuest(user) {
    let err;
    let guest = true;

    // logger.debug('this in schema statics', this);

    if (!this.roleExists(user)) {
      log.error(
        new Error(`user with role property
         must be provided in isGuest, defaulting to guest user`)
      );
    }

    const roleIndex = _roles.indexOf(user.role);

    if (roleIndex === -1) {
      err = new Error("user role doesn't exist");
    }

    if (roleIndex > 0) {
      guest = false;
    }

    // logger.error(err);
    return guest; // note that guest defaults to true if no roles found.
  };

  schema.statics.hasRole = function hasRole(userObj, minimumRoleRequired) {
    let user = userObj;
    let requiredRole = minimumRoleRequired;

    if (!user) {
      // logger.debug('no user supplied, defaulting to guest');
      user = { role: _roles[0] };
    }

    if (!requiredRole) {
      // logger.info('no required role supplied, defaulting to most strict');
      requiredRole = _roles[_roles.length - 1];
    }

    // logger.debug('requiredRole is', requiredRole);
    // default to most conservative role for safety
    let requiredRoleIndex = _roles.indexOf(requiredRole);
    // logger.debug('required role index is', requiredRoleIndex);
    if (requiredRoleIndex === -1) {
      // logger.debug(`required role doesn\'t exist,
      //   defaulting to highest to be conservative`);
      requiredRoleIndex = _roles.length - 1;
    }

    const userRole = user && user.role;
    // logger.debug('user role is', userRole);
    let index = _roles.indexOf(userRole);
    // logger.debug('user role index is', index);
    if (index === -1) {
      index = 0;
    }

    if (index < requiredRoleIndex) {
      return false;
    }

    return true;
  };
}
