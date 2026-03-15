{{-- partials/delete-user-form.blade.php --}}
<section class="space-y-6">
    <header class="profile-section-header">
        <h2 class="text-lg font-medium text-white">
            {{ __('Delete Account') }}
        </h2>

        <p class="mt-1 text-light">
            {{ __('Once your account is deleted, all of its resources and data will be permanently deleted. Before deleting your account, please download any data or information that you wish to retain.') }}
        </p>
    </header>

    <x-danger-button
        x-data=""
        x-on:click.prevent="$dispatch('open-modal', 'confirm-user-deletion')"
        class="btn-delete-account"
    >
        <i class="fas fa-trash me-2"></i>{{ __('Delete Account') }}
    </x-danger-button>

    <x-modal name="confirm-user-deletion" :show="$errors->userDeletion->isNotEmpty()" focusable>
        <form method="post" action="{{ route('profile.destroy') }}" class="p-6">
            @csrf
            @method('delete')

            <div class="modal-header-custom">
                <h2 class="text-lg font-medium text-white">
                    <i class="fas fa-exclamation-triangle me-2 text-warning"></i>
                    {{ __('Are you sure you want to delete your account?') }}
                </h2>
            </div>

            <p class="mt-1 text-light">
                {{ __('Once your account is deleted, all of its resources and data will be permanently deleted. Please enter your password to confirm you would like to permanently delete your account.') }}
            </p>

            <div class="mt-6">
                <x-input-label for="password" value="{{ __('Password') }}" class="sr-only" />

                <x-text-input
                    id="password"
                    name="password"
                    type="password"
                    class="mt-1 block w-3/4 form-input-custom"
                    placeholder="{{ __('Password') }}"
                />

                <x-input-error :messages="$errors->userDeletion->get('password')" class="mt-2" />
            </div>

            <div class="mt-6 flex justify-end gap-3">
                <x-secondary-button x-on:click="$dispatch('close')" class="btn-cancel-delete">
                    <i class="fas fa-times me-2"></i>{{ __('Cancel') }}
                </x-secondary-button>

                <x-danger-button class="btn-confirm-delete">
                    <i class="fas fa-trash me-2"></i>{{ __('Delete Account') }}
                </x-danger-button>
            </div>
        </form>
    </x-modal>
</section>